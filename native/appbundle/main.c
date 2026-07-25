/*
 * Stenograf.app/Contents/MacOS/Stenograf — the launcher stub.
 *
 * FROZEN. Read native/appbundle/README.md before changing a byte of this file.
 * TCC stores the app's microphone and system-audio grants against the *cdhash*
 * of this executable and nothing else (no identifier, no anchor, measured in
 * PLAN.md Phase 8 step 2), so any change here silently revokes the grant of
 * everyone who already answered the prompt. That is why the compiled binary is
 * committed rather than built at install time, and why everything that could
 * ever need to change — which program to launch, with which arguments — is read
 * from a file *outside* the bundle at every launch.
 *
 * Two hard rules follow from the same measurement:
 *
 *   1. This must be a Mach-O, not a script. A `#!/bin/sh` main executable makes
 *      the process launchd started *become* the interpreter, which lives outside
 *      the bundle, and TCC then path-keys the grant to the shared uv python3.13
 *      instead of to us.
 *   2. It must spawn the real program as a *child* and stay alive as its parent.
 *      An exec() throws the bundle identity away exactly like rule 1. Children
 *      inherit the responsible process, so the capture helper's prompt is
 *      attributed to "Stenograf" and its grant survives every reinstall of the
 *      Python side.
 *
 * The child is a GUI process with nowhere to write, so its output is teed into
 * ~/Library/Logs/Stenograf.log and a non-zero exit raises an alert naming it —
 * without that, a failed launch from the Dock is completely silent.
 */

#include <CoreFoundation/CoreFoundation.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

/* Relative to $HOME, all three deliberately fixed: this process is started by
 * launchd with no user environment, so an $XDG_/$STENOGRAF_ override read by
 * the Python side would name a file we could never find. */
#define LAUNCH_TARGET "Library/Application Support/stenograf/launch-target"
#define FALLBACK_PROGRAM ".local/bin/steno"
#define LOG_FILE "Library/Logs/Stenograf.log"

#define MAX_ARGV 64
#define MAX_TARGET_FILE 8192
#define LOG_LIMIT (1 << 20) /* rotate by truncation past 1 MiB */

static volatile sig_atomic_t child_pid = 0;

/* Quit/logout kills us, not the child, and an orphaned window with no parent
 * is worse than none — pass the signal on and let the child shut down; waitpid
 * then returns and we exit with it.
 *
 * Once there is no child left the handler must get out of the way: it is still
 * installed while an alert is on screen, and a handler that only forwards would
 * swallow the signal and leave a process nothing but SIGKILL can stop. */
static void forward_signal(int signal_number) {
    if (child_pid > 0) {
        kill(child_pid, signal_number);
        return;
    }
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

static void alert(const char *header, const char *message) {
    CFStringRef header_ref = CFStringCreateWithCString(NULL, header, kCFStringEncodingUTF8);
    CFStringRef message_ref = CFStringCreateWithCString(NULL, message, kCFStringEncodingUTF8);
    CFOptionFlags response = 0;
    /* Drawn by the system, so it appears even though this process never
     * connects to the window server (LSUIElement, see Info.plist). */
    CFUserNotificationDisplayAlert(0.0, kCFUserNotificationStopAlertLevel, NULL, NULL, NULL,
                                   header_ref, message_ref, NULL, NULL, NULL, &response);
    if (header_ref) CFRelease(header_ref);
    if (message_ref) CFRelease(message_ref);
}

static const char *home_dir(void) {
    const char *home = getenv("HOME");
    return (home != NULL && home[0] == '/') ? home : NULL;
}

/* $HOME/<relative> into buf; false when it does not fit or $HOME is unusable. */
static int home_path(char *buf, size_t size, const char *relative) {
    const char *home = home_dir();
    if (home == NULL) return 0;
    int written = snprintf(buf, size, "%s/%s", home, relative);
    return written > 0 && (size_t)written < size;
}

static char *trim(char *text) {
    while (*text == ' ' || *text == '\t') text++;
    size_t length = strlen(text);
    while (length > 0) {
        char last = text[length - 1];
        if (last != ' ' && last != '\t' && last != '\r') break;
        text[--length] = '\0';
    }
    return text;
}

/*
 * Parse the launch-target file: one argv element per line, first line the
 * program. Blank lines and #-comments are skipped, a leading ~/ is expanded.
 * Returns the number of elements written (0 when the file is missing or empty),
 * with the pointers aiming into `storage`, which the caller keeps alive.
 */
static int read_launch_target(const char *path, char *storage, size_t storage_size, char **argv,
                              int max_argv) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    ssize_t got = read(fd, storage, storage_size - 1);
    close(fd);
    if (got <= 0) return 0;
    storage[got] = '\0';

    int count = 0;
    char *cursor = storage;
    while (cursor != NULL && *cursor != '\0' && count < max_argv) {
        char *line = cursor;
        char *newline = strchr(cursor, '\n');
        if (newline != NULL) {
            *newline = '\0';
            cursor = newline + 1;
        } else {
            cursor = NULL;
        }
        line = trim(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        argv[count++] = line;
    }
    return count;
}

/* A hand-edited launch-target file may say ~/... — expand it for the program. */
static const char *expand_home(const char *path, char *buf, size_t size) {
    if (path[0] != '~' || path[1] != '/') return path;
    if (!home_path(buf, size, path + 2)) return path;
    return buf;
}

/*
 * ~/Library/Logs/Stenograf.log, opened for the child's stdout and stderr.
 * Truncates once past LOG_LIMIT rather than rotating: this is a diagnostic of
 * last resort, and the interesting run is always the most recent one.
 */
static int open_log(void) {
    char path[PATH_MAX];
    if (!home_path(path, sizeof path, LOG_FILE)) return -1;
    struct stat info;
    int flags = O_WRONLY | O_CREAT | O_APPEND;
    if (stat(path, &info) == 0 && info.st_size > LOG_LIMIT) flags = O_WRONLY | O_CREAT | O_TRUNC;
    int fd = open(path, flags, 0600);
    if (fd < 0) return -1;

    char stamp[32] = "";
    time_t now = time(NULL);
    struct tm local;
    if (localtime_r(&now, &local) != NULL) {
        strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &local);
    }
    dprintf(fd, "\n--- Stenograf.app launch %s ---\n", stamp);
    return fd;
}

static void note_bundle_path(void) {
    char executable[PATH_MAX];
    uint32_t size = sizeof executable;
    if (_NSGetExecutablePath(executable, &size) != 0) return;
    /* .../Stenograf.app/Contents/MacOS/Stenograf -> .../Stenograf.app */
    for (int level = 0; level < 3; level++) {
        char *slash = strrchr(executable, '/');
        if (slash == NULL) return;
        *slash = '\0';
    }
    /* Lets the Python side tell an app launch from a terminal launch — which is
     * the difference between "the app holds the TCC grant" and "your terminal
     * does". Nothing depends on it; it is a breadcrumb for `steno doctor`. */
    setenv("STENOGRAF_APP_BUNDLE", executable, 1);
}

/*
 * launchd hands a GUI app a bare PATH (/usr/bin:/bin:/usr/sbin:/sbin), so tools
 * a terminal session takes for granted — a user-installed ollama, a Homebrew
 * binary — are invisible to the child. Appended, never prepended: the system
 * copies of everything still win.
 */
static void widen_path(void) {
    const char *home = home_dir();
    if (home == NULL) return;
    const char *current = getenv("PATH");
    char widened[4096];
    int written = snprintf(widened, sizeof widened, "%s%s%s/.local/bin:/opt/homebrew/bin:/usr/local/bin",
                           current != NULL ? current : "", current != NULL && *current ? ":" : "",
                           home);
    if (written > 0 && (size_t)written < sizeof widened) setenv("PATH", widened, 1);
}

int main(int argc, char *argv[]) {
    char storage[MAX_TARGET_FILE];
    char *child_argv[MAX_ARGV];
    int count = 0;

    char target_file[PATH_MAX];
    if (home_path(target_file, sizeof target_file, LAUNCH_TARGET)) {
        count = read_launch_target(target_file, storage, sizeof storage, child_argv, MAX_ARGV - 8);
    }

    char fallback[PATH_MAX];
    if (count == 0) {
        /* No file yet (or an unreadable one): the default install location,
         * which is also what `steno setup` would have written there. */
        if (!home_path(fallback, sizeof fallback, FALLBACK_PROGRAM)) {
            alert("Stenograf could not start", "The home folder could not be determined.");
            return 1;
        }
        child_argv[count++] = fallback;
    }

    char program[PATH_MAX];
    child_argv[0] = (char *)expand_home(child_argv[0], program, sizeof program);

    /* A file naming only a program means the desktop app: `steno --gui`. The
     * flag is frozen into this binary, so the CLI must keep accepting it even
     * after the GUI becomes the default. */
    if (count == 1) child_argv[count++] = "--gui";

    /* Anything from `open -a Stenograf --args …` is passed through, minus the
     * legacy process-serial-number marker LaunchServices still sometimes adds. */
    for (int i = 1; i < argc && count < MAX_ARGV - 1; i++) {
        if (strncmp(argv[i], "-psn_", 5) != 0) child_argv[count++] = argv[i];
    }
    child_argv[count] = NULL;

    if (access(child_argv[0], X_OK) != 0) {
        char message[PATH_MAX + 256];
        snprintf(message, sizeof message,
                 "%s cannot be run.\n\nStenograf is installed separately from this app. "
                 "Install it (see the project's README), then run `steno setup` to point "
                 "this app at it.",
                 child_argv[0]);
        alert("Stenograf is not installed", message);
        return 1;
    }

    note_bundle_path();
    widen_path();

    int log_fd = open_log();
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (log_fd >= 0) {
        posix_spawn_file_actions_adddup2(&actions, log_fd, STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&actions, log_fd, STDERR_FILENO);
    }

    pid_t pid = 0;
    /* posix_spawn, never exec: rule 2 at the top of this file. */
    int spawn_error = posix_spawn(&pid, child_argv[0], &actions, NULL, child_argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawn_error != 0) {
        char message[PATH_MAX + 256];
        snprintf(message, sizeof message, "%s could not be started: %s", child_argv[0],
                 strerror(spawn_error));
        alert("Stenograf could not start", message);
        return 1;
    }

    child_pid = pid;
    signal(SIGTERM, forward_signal);
    signal(SIGINT, forward_signal);
    signal(SIGHUP, forward_signal);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            /* Nothing left to wait on; treat it as a clean detach rather than
             * accusing the app of crashing. */
            return 0;
        }
    }
    child_pid = 0; /* forward_signal stops forwarding and starts obeying */
    if (log_fd >= 0) close(log_fd);

    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        char message[512];
        snprintf(message, sizeof message,
                 "Stenograf stopped with error %d.\n\nThe details are in "
                 "~/Library/Logs/Stenograf.log — or run `steno --gui` in a terminal to see "
                 "the same message directly.",
                 WEXITSTATUS(status));
        alert("Stenograf quit unexpectedly", message);
        return WEXITSTATUS(status);
    }
    /* A signalled child is either our own forwarded quit or a crash macOS has
     * already reported; either way a second dialog adds nothing. */
    return 0;
}

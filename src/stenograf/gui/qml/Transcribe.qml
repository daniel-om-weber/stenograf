// Transcribe a recording: pick a file, run the finalize pipeline, watch it.
// A native file dialog rather than a tree of our own — the desktop already has
// a good one, and it is the part of this screen we would otherwise maintain.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import QtQuick.Layouts

Panel {
    id: page

    property var screen

    heading: "Transcribe a recording"
    hint: "Pick an audio or video file; its audio track is transcribed into a new meeting folder."
    busy: page.screen.state.busy

    FileDialog {
        id: picker

        title: "Choose a recording"
        currentFolder: page.screen.state.home
        nameFilters: ["Audio and video (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.aiff *.aif *.wma *.mka *.mp4 *.mov *.webm *.mkv)", "All files (*)"]
        onAccepted: page.screen.choose(picker.selectedFile.toString())
    }

    RowLayout {
        spacing: 12
        Layout.fillWidth: true

        Btn {
            text: "Choose file…"
            enabled: !page.screen.state.busy
            onClicked: picker.open()
        }

        Text {
            text: page.screen.state.file || "No file selected."
            color: page.screen.state.file ? Theme.text : Theme.muted
            font.pixelSize: 13
            elide: Text.ElideMiddle
            Layout.fillWidth: true
        }
    }

    Rectangle {
        visible: page.screen.state.busy
        height: 6
        radius: 3
        color: Theme.surfaceHi
        Layout.fillWidth: true
        Layout.topMargin: 6

        Rectangle {
            width: parent.width * page.screen.state.progress
            height: parent.height
            radius: 3
            color: Theme.accent
        }
    }

    Text {
        text: page.screen.state.status
        visible: text.length > 0
        color: Theme.muted
        font.pixelSize: 12
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
        Layout.topMargin: 4
    }

    RowLayout {
        spacing: 10
        Layout.fillWidth: true
        Layout.topMargin: 8

        Btn {
            text: "Transcribe"
            primary: true
            enabled: !page.screen.state.busy && page.screen.state.file.length > 0
            Layout.fillWidth: true
            onClicked: page.screen.start()
        }

        Btn {
            // Leaving mid-run is refused rather than silently allowed: the
            // worker cannot be interrupted safely.
            text: "Back"
            enabled: !page.screen.state.busy
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    }
}

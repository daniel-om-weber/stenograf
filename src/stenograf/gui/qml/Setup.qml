// Meeting setup: the few choices that matter before capture starts. One concept
// per control; the per-channel counts appear only while diarization is on,
// because that is the only time they mean anything. Everything the form does
// not ask about comes from settings.toml, exactly as for a flagless
// `steno start` — resolving the two together is the library's job, not this
// file's, so Start hands the raw controls over and shows whatever comes back.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Panel {
    id: page

    property var screen

    heading: "Start meeting"
    hint: "Formats, vocabulary and the rest come from your settings."

    // The standing defaults are copied into the switches once, after opened()
    // refreshed them — never bound: the microphone listing lands a moment
    // later through the same state map, and a binding re-evaluated then would
    // flip back a switch the user had already changed.
    Component.onCompleted: {
        if (!page.screen)
            return;
        page.screen.opened();
        diarize.checked = page.screen.state.diarize;
        record.checked = page.screen.state.recordAudio;
        notes.checked = page.screen.state.notes;
    }

    // The meeting type comes first because it sets defaults for the controls
    // below it — and it is hidden entirely on a machine whose settings.toml
    // defines no [meetings.*] section, so the form nobody has presets for looks
    // exactly as it did before this control existed.
    ColumnLayout {
        spacing: 6
        visible: page.screen.state.presets.length > 1
        Layout.fillWidth: true
        Layout.bottomMargin: 4  // the summary must not crowd the first switch

        Text {
            text: "Meeting type"
            color: Theme.text
            font.pixelSize: Theme.textBody
        }

        Combo {
            id: preset

            model: page.screen.state.presets
            Layout.fillWidth: true
        }

        Text {
            text: preset.currentOption ? (preset.currentOption.hint || "") : ""
            visible: text.length > 0
            color: Theme.muted
            font.pixelSize: Theme.textSmall
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    Toggle {
        id: mic

        checked: true
        label: "Microphone"
        hint: mic.checked ? "People in the room." : "Starts muted — switch it on in the meeting."
    }

    // Which microphone, under its switch — shown either way, because a mic that
    // starts muted is still recorded and can be switched on. The combo appears
    // only once Python has listed the devices (the model arrives from a helper
    // subprocess, and a model swap resets the selection) and only when there is
    // something to choose between; the hint below it is separate, because a
    // listing that failed must say so rather than show nothing at all.
    ColumnLayout {
        spacing: 6
        Layout.fillWidth: true

        Text {
            text: "Record from"
            visible: micDevice.visible
            color: Theme.text
            font.pixelSize: Theme.textBody
        }

        Combo {
            id: micDevice

            model: page.screen.state.micDevices
            visible: page.screen.state.micDevicesReady && page.screen.state.micDevices.length > 1
            Layout.fillWidth: true
        }

        Text {
            text: page.screen.state.micError
            visible: text.length > 0
            color: Theme.muted
            font.pixelSize: Theme.textSmall
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    Toggle {
        id: system

        checked: true
        label: "System audio"
        hint: system.checked ? "Calls, videos — anything playing on this machine." : "Starts muted — switch it on in the meeting."
    }

    Toggle {
        id: diarize

        label: "Tell speakers apart"
        hint: "Off: each source is one speaker in the transcript."
    }

    ColumnLayout {
        spacing: 8
        visible: diarize.checked
        Layout.fillWidth: true
        Layout.topMargin: 4

        Text {
            text: "Speakers in the room (microphone)"
            color: Theme.text
            font.pixelSize: Theme.textBody
        }

        Combo {
            id: localCount

            model: page.screen.counts
            Layout.fillWidth: true
        }

        Text {
            text: "Remote speakers (system audio)"
            color: Theme.text
            font.pixelSize: Theme.textBody
        }

        Combo {
            id: remoteCount

            model: page.screen.counts
            Layout.fillWidth: true
        }

        Text {
            text: "Auto-detect counts speakers well; an exact count drops stray voices."
            color: Theme.muted
            font.pixelSize: Theme.textSmall
        }
    }

    Text {
        text: "Title (optional; used by notes)"
        color: Theme.text
        font.pixelSize: Theme.textBody
        Layout.topMargin: 4
    }

    Field {
        id: title

        placeholderText: "e.g. Weekly sync"
        Layout.fillWidth: true
    }

    Toggle {
        id: record

        label: "Keep the audio recording"
        hint: "Writes audio.opus next to the transcript."
    }

    Toggle {
        id: notes

        label: "Generate notes afterwards"
    }

    Text {
        text: page.screen.state.error
        visible: text.length > 0
        color: Theme.bad
        font.pixelSize: Theme.textBody
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
        Layout.topMargin: 4
    }

    actions: [
        Btn {
            text: "Start"
            primary: true
            Layout.fillWidth: true
            onClicked: page.screen.start({
                "preset": preset.value || "",
                // || "": an empty model yields undefined, which crosses to
                // Python as a missing key rather than "no device chosen".
                "micDevice": micDevice.value || "",
                "mic": mic.checked,
                "system": system.checked,
                "diarize": diarize.checked,
                "local": localCount.value,
                "remote": remoteCount.value,
                "title": title.text,
                "recordAudio": record.checked,
                "notes": notes.checked
            })
        },

        Btn {
            text: "Back"
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    ]
}

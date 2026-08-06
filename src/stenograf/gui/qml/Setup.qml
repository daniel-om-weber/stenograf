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

    Component.onCompleted: if (page.screen)
        page.screen.opened()

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
            font.pixelSize: 13
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
            font.pixelSize: 12
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    Toggle {
        id: mic

        checked: true
        label: "Microphone"
        hint: "People in the room."
    }

    Toggle {
        id: system

        checked: true
        label: "System audio"
        hint: "Calls, videos — anything playing on this machine."
    }

    Toggle {
        id: diarize

        checked: page.screen.state.diarize
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
            font.pixelSize: 13
        }

        Combo {
            id: localCount

            model: page.screen.counts
            Layout.fillWidth: true
        }

        Text {
            text: "Remote speakers (system audio)"
            color: Theme.text
            font.pixelSize: 13
        }

        Combo {
            id: remoteCount

            model: page.screen.counts
            Layout.fillWidth: true
        }

        Text {
            text: "Auto-detect works; exact counts label speakers better."
            color: Theme.muted
            font.pixelSize: 12
        }
    }

    Text {
        text: "Language"
        color: Theme.text
        font.pixelSize: 13
        Layout.topMargin: 4
    }

    Combo {
        id: language

        model: page.screen.languages
        Layout.fillWidth: true
    }

    Text {
        text: "Title (optional; used by notes)"
        color: Theme.text
        font.pixelSize: 13
        Layout.topMargin: 4
    }

    Field {
        id: title

        placeholderText: "e.g. Weekly sync"
        Layout.fillWidth: true
    }

    Toggle {
        id: record

        checked: page.screen.state.recordAudio
        label: "Keep the audio recording"
        hint: "Writes audio.opus next to the transcript."
    }

    Toggle {
        id: notes

        checked: page.screen.state.notes
        label: "Generate notes afterwards"
    }

    Text {
        text: page.screen.state.error
        visible: text.length > 0
        color: Theme.bad
        font.pixelSize: 13
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
        Layout.topMargin: 4
    }

    RowLayout {
        spacing: 10
        Layout.fillWidth: true
        Layout.topMargin: 8

        Btn {
            text: "Start"
            primary: true
            Layout.fillWidth: true
            onClicked: page.screen.start({
                "preset": preset.value || "",
                "mic": mic.checked,
                "system": system.checked,
                "diarize": diarize.checked,
                "local": localCount.value,
                "remote": remoteCount.value,
                "language": language.value,
                "title": title.text,
                "recordAudio": record.checked,
                "notes": notes.checked
            })
        }

        Btn {
            text: "Back"
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    }
}

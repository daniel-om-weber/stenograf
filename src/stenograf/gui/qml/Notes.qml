// Generate notes for a finished meeting. The newest one is pre-selected — that
// is what `steno notes --last` does, and most notes runs happen right after the
// meeting they summarize — and anything else is a folder you pick.
//
// This stays a dumb picker: no titles, no dates, no summaries. A list of
// meetings with metadata would be the meeting browser the product forbids.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import QtQuick.Layouts

Panel {
    id: page

    property var screen

    heading: "Generate notes"
    hint: "Summarizes a meeting's transcript into notes.md next to it."
    busy: page.screen.state.busy

    FolderDialog {
        id: picker

        title: "Choose a meeting folder"
        currentFolder: page.screen.state.home
        onAccepted: page.screen.choose(picker.selectedFolder.toString())
    }

    RowLayout {
        spacing: 12
        Layout.fillWidth: true

        Btn {
            text: "Choose meeting…"
            enabled: !page.screen.state.busy
            onClicked: picker.open()
        }

        Text {
            text: page.screen.state.meeting || "No meeting selected."
            color: page.screen.state.meeting ? Theme.text : Theme.muted
            font.pixelSize: 13
            elide: Text.ElideMiddle
            Layout.fillWidth: true
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
            text: "Generate notes"
            primary: true
            enabled: !page.screen.state.busy && page.screen.state.meeting.length > 0
            Layout.fillWidth: true
            onClicked: page.screen.start()
        }

        Btn {
            text: "Back"
            enabled: !page.screen.state.busy
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    }
}

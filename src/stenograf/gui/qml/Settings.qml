// The effective configuration, read-only, with every value's provenance. Not a
// form: editing happens in settings.toml itself, which the Open button hands to
// whatever this desktop uses for .toml files.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Panel {
    id: page

    property var screen

    heading: "Settings"
    hint: "Where each value comes from: an environment override, a meeting type, settings.toml, or the built-in default."
    cardWidth: 720

    Component.onCompleted: if (page.screen)
        page.screen.opened()

    // Not an editor: picking a meeting type re-renders the same read-only report
    // with that [meetings.*] overlay applied, which is the only way to see what
    // a type actually changes. Hidden when no presets are defined.
    RowLayout {
        spacing: 10
        visible: page.screen.state.presets.length > 1
        Layout.fillWidth: true
        Layout.bottomMargin: 4

        Text {
            text: "Meeting type"
            color: Theme.text
            font.pixelSize: 13
        }

        Combo {
            id: preset

            model: page.screen.state.presets
            // Driven by the screen's own selection, so Reload (which re-reads
            // the file and may drop a deleted preset) and the report can never
            // disagree about which type is on display.
            currentIndex: Math.max(0, preset.model.findIndex(entry => entry.value === page.screen.state.preset))
            Layout.fillWidth: true
            onActivated: page.screen.show(preset.value || "")
        }
    }

    // Selectable, so a path or a value can be copied out; wrapped, because a
    // long glossary path or attendee list would otherwise run off the card.
    TextEdit {
        text: page.screen.state.text
        color: page.screen.state.ok ? Theme.text : Theme.bad
        font.pixelSize: 12
        font.family: Theme.mono
        wrapMode: Text.Wrap
        readOnly: true
        selectByMouse: true
        selectionColor: Theme.accent
        selectedTextColor: Theme.accentText
        Layout.fillWidth: true
    }

    RowLayout {
        spacing: 10
        Layout.fillWidth: true
        Layout.topMargin: 10

        Btn {
            text: "Open settings.toml"
            Layout.fillWidth: true
            onClicked: page.screen.edit()
        }

        Btn {
            text: "Reload"
            Layout.fillWidth: true
            onClicked: page.screen.opened()
        }

        Btn {
            text: "Back"
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    }
}

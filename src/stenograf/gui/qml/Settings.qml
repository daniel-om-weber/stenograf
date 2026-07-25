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
    hint: "Where each value comes from: an environment override, settings.toml, or the built-in default."
    cardWidth: 720

    Component.onCompleted: if (page.screen)
        page.screen.opened()

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

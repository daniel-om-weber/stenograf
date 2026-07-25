// A single-line text input in the app's palette.
import QtQuick
import QtQuick.Controls.Basic

TextField {
    id: control

    implicitHeight: 36
    leftPadding: 12
    rightPadding: 12
    font.pixelSize: 13
    color: Theme.text
    placeholderTextColor: Theme.dim
    selectionColor: Theme.accent
    selectedTextColor: Theme.accentText

    background: Rectangle {
        radius: 8
        color: Theme.surfaceHi
        border.width: 1
        border.color: control.activeFocus ? Theme.accent : Theme.line
    }
}

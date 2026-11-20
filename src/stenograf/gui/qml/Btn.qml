// A button in the app's palette. Hover animates because it is event-driven and
// stops; nothing here runs while the app sits idle.
import QtQuick
import QtQuick.Controls.Basic

Button {
    id: control

    property bool primary: false
    property string glyph: ""  // an Icon name shown before the text; not `icon`, which Button owns
    // `flat` (Button's own property) is the quiet variant: no fill or border
    // until hovered, muted ink — for a way out that should not compete.
    readonly property color ink: !control.enabled ? Theme.dim : control.primary ? Theme.accentText : control.flat && !control.hovered ? Theme.muted : Theme.text

    implicitHeight: 36
    leftPadding: 16
    rightPadding: 16
    font.pixelSize: Theme.textBody

    background: Rectangle {
        radius: 8
        color: control.flat ? (control.hovered ? Theme.surfaceHi : "transparent") : !control.enabled ? Theme.surfaceHi : control.primary ? (control.hovered ? Qt.lighter(Theme.accent, 1.12) : Theme.accent) : (control.hovered ? Theme.surfaceHi : Theme.control)
        border.width: control.primary || (control.flat && !control.activeFocus) ? 0 : 1
        border.color: control.activeFocus ? Theme.accent : Theme.line

        Behavior on color {
            ColorAnimation {
                duration: 90
            }
        }
    }

    contentItem: Item {
        // From the label's natural width, not the row's: the label's own width
        // is clamped to the button, so deriving the button from it would loop.
        implicitWidth: label.implicitWidth + (control.glyph !== "" ? 22 : 0)
        implicitHeight: label.implicitHeight

        Row {
            id: row

            anchors.centerIn: parent
            spacing: 7

            Icon {
                anchors.verticalCenter: parent.verticalCenter
                visible: control.glyph !== ""
                name: control.glyph
                color: control.ink
                size: 15
            }

            Text {
                id: label

                anchors.verticalCenter: parent.verticalCenter
                width: Math.min(implicitWidth, control.availableWidth - (control.glyph !== "" ? 22 : 0))
                text: control.text
                font: control.font
                color: control.ink
                elide: Text.ElideRight
            }
        }
    }
}

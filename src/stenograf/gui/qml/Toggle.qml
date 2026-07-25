// A switch with its label and (optionally) a line of explanation — the setup
// form's unit of one concept, one control.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

RowLayout {
    id: row

    property alias checked: control.checked
    property string label: ""
    property string hint: ""

    spacing: 12
    Layout.fillWidth: true

    Switch {
        id: control

        implicitWidth: 44
        implicitHeight: 24
        padding: 0
        Layout.alignment: Qt.AlignTop

        indicator: Rectangle {
            implicitWidth: 44
            implicitHeight: 24
            radius: 12
            color: control.checked ? Theme.accent : Theme.surfaceHi
            border.width: 1
            border.color: control.checked ? Theme.accent : Theme.line

            Behavior on color {
                ColorAnimation {
                    duration: 90
                }
            }

            Rectangle {
                x: control.checked ? parent.width - width - 3 : 3
                y: 3
                width: 18
                height: 18
                radius: 9
                color: control.checked ? Theme.accentText : Theme.muted

                Behavior on x {
                    NumberAnimation {
                        duration: 90
                    }
                }
            }
        }

        contentItem: Item {}
    }

    ColumnLayout {
        spacing: 2
        Layout.fillWidth: true

        Text {
            text: row.label
            color: Theme.text
            font.pixelSize: 14
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }

        Text {
            text: row.hint
            visible: row.hint.length > 0
            color: Theme.muted
            font.pixelSize: 12
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }
}

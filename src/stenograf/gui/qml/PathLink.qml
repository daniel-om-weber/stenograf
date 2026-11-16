// A folder path that opens in the file manager when clicked. It stays dim until
// hovered, so a footer path reads as information first and a link second.
import QtQuick
import QtQuick.Controls.Basic

Item {
    id: link

    property alias text: label.text
    readonly property color ink: area.containsMouse ? Theme.accent : Theme.dim

    signal activated

    implicitWidth: glyph.width + 6 + label.implicitWidth
    implicitHeight: Math.max(glyph.height, label.implicitHeight)
    Accessible.role: Accessible.Link
    Accessible.name: "Open folder " + text
    Accessible.onPressAction: link.activated()

    Icon {
        id: glyph

        anchors.verticalCenter: parent.verticalCenter
        name: "folder"
        color: link.ink
        size: 13
    }

    Text {
        id: label

        anchors {
            left: glyph.right
            leftMargin: 6
            right: parent.right
            verticalCenter: parent.verticalCenter
        }
        color: link.ink
        font.pixelSize: 11
        font.underline: area.containsMouse
        elide: Text.ElideMiddle
    }

    MouseArea {
        id: area

        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: link.activated()
    }

    ToolTip.visible: area.containsMouse
    ToolTip.delay: 600
    ToolTip.text: "Open folder"
}

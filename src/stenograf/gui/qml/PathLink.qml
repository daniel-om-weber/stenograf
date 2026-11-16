// A folder path that opens in the file manager when clicked. It stays dim until
// hovered, so a footer path reads as information first and a link second.
import QtQuick
import QtQuick.Controls.Basic

Text {
    id: link

    signal activated

    color: area.containsMouse ? Theme.accent : Theme.dim
    font.pixelSize: 11
    font.underline: area.containsMouse
    elide: Text.ElideMiddle
    Accessible.role: Accessible.Link
    Accessible.name: "Open folder " + text
    Accessible.onPressAction: link.activated()

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

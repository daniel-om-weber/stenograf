// The shape every non-meeting screen has: a centred card with a heading, an
// optional hint, and whatever the page puts inside it. The card scrolls rather
// than clipping when the window is shorter than its content, and Escape leaves
// — unless the page says it is busy, because a worker thread cannot be
// interrupted safely and finishing behind a closed page would surprise more.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: panel

    property var app
    property string heading: ""
    property string hint: ""
    property bool busy: false
    property int cardWidth: 620

    default property alias content: column.data

    Shortcut {
        sequences: [StandardKey.Cancel]
        onActivated: if (!panel.busy && panel.app)
            panel.app.back()
    }

    Flickable {
        id: flick

        anchors.centerIn: parent
        width: Math.min(panel.cardWidth, panel.width - 56)
        height: Math.min(card.height, panel.height - 48)
        contentWidth: width
        contentHeight: card.height
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar {}

        Rectangle {
            id: card

            width: flick.width
            height: column.implicitHeight + 44
            radius: 14
            color: Theme.surface
            border.width: 1
            border.color: Theme.line

            ColumnLayout {
                id: column

                x: 22
                y: 22
                width: parent.width - 44
                spacing: 10

                Text {
                    text: panel.heading
                    color: Theme.text
                    font.pixelSize: 20
                    font.weight: Font.Medium
                    Layout.fillWidth: true
                }

                Text {
                    text: panel.hint
                    visible: panel.hint.length > 0
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    Layout.bottomMargin: 6
                }
            }
        }
    }
}

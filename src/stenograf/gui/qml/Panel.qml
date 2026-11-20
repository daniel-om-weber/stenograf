// The shape every non-meeting screen has: a centred card with a heading, an
// optional hint, whatever the page puts inside it, and a row of actions. The
// body scrolls when the window is shorter than the card; the actions do not —
// a Start below the fold is a form with no Start (measured at 960x500,
// 2026-11-20, the scaling a 7-inch 1080p panel gets at 200 %). Escape leaves —
// unless the page says it is busy, because a worker thread cannot be
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

    // Set to accept a file dropped anywhere on the page; the text says what
    // the drop will do, over the page while something is dragged across it.
    property string dropHint: ""

    signal dropped(string url)

    default property alias content: column.data
    // The buttons pinned under the scrolling body, laid out in a row.
    property alias actions: actionRow.data

    Shortcut {
        sequences: [StandardKey.Cancel]
        onActivated: if (!panel.busy && panel.app)
            panel.app.back()
    }

    DropArea {
        id: dropZone

        anchors.fill: parent
        enabled: panel.dropHint.length > 0 && !panel.busy
        onEntered: drag => drag.accepted = drag.hasUrls
        onDropped: drop => {
            if (drop.hasUrls && drop.urls.length > 0) {
                drop.acceptProposedAction();
                panel.dropped(drop.urls[0].toString());
            }
        }
    }

    Rectangle {
        id: card

        readonly property int pad: 22
        readonly property int footer: actionRow.children.length > 0 ? actionRow.implicitHeight + card.pad : 0

        anchors.centerIn: parent
        width: Math.min(panel.cardWidth, panel.width - 32)
        height: Math.min(column.implicitHeight + 2 * card.pad + card.footer, panel.height - 32)
        radius: 14
        color: Theme.surface
        border.width: 1
        border.color: Theme.line

        Flickable {
            id: flick

            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
                bottom: actionRow.top
                topMargin: card.pad
                // The scrollbar sits in the card's padding, clear of the text.
                rightMargin: 6
                bottomMargin: actionRow.children.length > 0 ? card.pad : 0
            }
            contentWidth: width
            contentHeight: column.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            // Basic's scrollbar shows only while scrolling, which makes a
            // clipped body look complete.
            ScrollBar.vertical: ScrollBar {
                policy: flick.contentHeight > flick.height ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
            }

            ColumnLayout {
                id: column

                x: card.pad
                width: card.width - 2 * card.pad
                spacing: 10

                Text {
                    text: panel.heading
                    color: Theme.text
                    font.pixelSize: Theme.textHeading
                    font.weight: Font.Medium
                    Layout.fillWidth: true
                }

                Text {
                    text: panel.hint
                    visible: panel.hint.length > 0
                    color: Theme.muted
                    font.pixelSize: Theme.textBody
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    Layout.bottomMargin: 6
                }
            }
        }

        RowLayout {
            id: actionRow

            anchors {
                left: parent.left
                right: parent.right
                bottom: parent.bottom
                margins: card.pad
            }
            spacing: 10
        }
    }

    Rectangle {
        visible: dropZone.containsDrag
        anchors.fill: card
        radius: card.radius
        color: Qt.rgba(Theme.bg.r, Theme.bg.g, Theme.bg.b, 0.92)
        border.width: 2
        border.color: Theme.accent

        Column {
            anchors.centerIn: parent
            spacing: 10

            Icon {
                anchors.horizontalCenter: parent.horizontalCenter
                name: "upload"
                color: Theme.accent
                size: 28
            }

            Text {
                text: panel.dropHint
                color: Theme.text
                font.pixelSize: Theme.textLabel
            }
        }
    }
}

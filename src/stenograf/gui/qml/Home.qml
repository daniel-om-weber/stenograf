// The menu: one large labelled card per workflow, each carrying a line about
// what it does — the audience is people who don't know the subcommands. Cards
// are real Buttons, so the whole menu is keyboard-reachable (tab + space).
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page

    property var app

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(560, page.width - 80)
        spacing: 10

        Text {
            text: "stenograf"
            color: Theme.text
            font.pixelSize: 34
            font.weight: Font.Light
            font.letterSpacing: 1.5
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            text: "local meeting transcription"
            color: Theme.muted
            font.pixelSize: 14
            Layout.alignment: Qt.AlignHCenter
            Layout.bottomMargin: 18
        }

        Repeater {
            model: page.app.menu

            delegate: Button {
                id: card

                required property var modelData
                required property int index

                Layout.fillWidth: true
                implicitHeight: 64
                padding: 0
                focus: card.index === 0

                onClicked: card.modelData.page === "quit" ? Qt.quit() : page.app.open(card.modelData.page)

                background: Rectangle {
                    radius: 12
                    color: card.hovered ? Theme.surfaceHi : Theme.surface
                    border.width: 1
                    border.color: card.hovered || card.activeFocus ? Theme.accent : Theme.line

                    Behavior on color {
                        ColorAnimation {
                            duration: 90
                        }
                    }
                    Behavior on border.color {
                        ColorAnimation {
                            duration: 90
                        }
                    }
                }

                contentItem: ColumnLayout {
                    spacing: 3

                    Text {
                        text: card.modelData.label
                        color: Theme.text
                        font.pixelSize: 15
                        font.weight: Font.Medium
                        Layout.leftMargin: 18
                    }

                    Text {
                        text: card.modelData.description
                        color: Theme.muted
                        font.pixelSize: 12
                        Layout.leftMargin: 18
                    }
                }
            }
        }
    }
}

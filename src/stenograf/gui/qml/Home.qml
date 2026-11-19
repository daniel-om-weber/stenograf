// The menu: one large labelled card per workflow, each carrying a line about
// what it does — the audience is people who don't know the subcommands. Cards
// are real Buttons, so the whole menu is keyboard-reachable (tab + space).
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page

    property var app
    // The update line's object (gui/updates.py). Not a page, but fetched like
    // one, so this file needs no second injected property.
    readonly property var updates: page.app.screen("Update")

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

                // quit_app(), not Qt.quit(): leaving with a meeting running has
                // to stop capture and let the finalize land first.
                onClicked: card.modelData.page === "quit" ? page.app.quit_app() : page.app.open(card.modelData.page)

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

    // One line about updates, at the foot of the menu and only there: no
    // dialog, no notification, and nothing at all while a meeting runs
    // (`quiet`). It is the whole update interface besides the menu-bar entry.
    Rectangle {
        visible: page.updates.state.phase !== "none" && !page.updates.state.quiet
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 12
        width: Math.min(560, page.width - 80)
        height: updateRow.implicitHeight + 14
        radius: 10
        color: Theme.surface
        border.width: 1
        border.color: page.updates.state.phase === "failed" ? Theme.bad : Theme.line

        RowLayout {
            id: updateRow

            anchors.fill: parent
            anchors.leftMargin: 14
            anchors.rightMargin: 8
            spacing: 8

            ColumnLayout {
                spacing: 1
                Layout.fillWidth: true

                Text {
                    text: page.updates.state.text
                    color: Theme.text
                    font.pixelSize: 13
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }

                // Selectable: on an install that cannot update itself this is
                // the command to paste into a terminal.
                TextEdit {
                    visible: text !== ""
                    text: page.updates.state.detail
                    color: Theme.muted
                    font.pixelSize: 11
                    wrapMode: Text.Wrap
                    readOnly: true
                    selectByMouse: true
                    selectionColor: Theme.accent
                    selectedTextColor: Theme.accentText
                    Layout.fillWidth: true
                }
            }

            Btn {
                text: "What's new"
                implicitHeight: 28
                visible: ["available", "notice", "ready", "updated"].indexOf(page.updates.state.phase) >= 0
                onClicked: page.updates.open_notes()
            }

            Btn {
                text: "Update"
                primary: true
                implicitHeight: 28
                visible: page.updates.state.phase === "available"
                onClicked: page.updates.install()
            }

            Btn {
                text: "Restart now"
                primary: true
                implicitHeight: 28
                visible: page.updates.state.phase === "ready"
                onClicked: page.updates.restart()
            }

            Btn {
                text: "Cancel"
                implicitHeight: 28
                visible: page.updates.state.phase === "preparing"
                onClicked: page.updates.cancel()
            }

            Btn {
                text: page.updates.state.phase === "available" ? "Later" : "Hide"
                implicitHeight: 28
                visible: ["available", "notice", "updated", "failed"].indexOf(page.updates.state.phase) >= 0
                onClicked: page.updates.dismiss()
            }
        }
    }
}

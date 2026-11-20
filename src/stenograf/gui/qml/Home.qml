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

    // Short windows (a 7-inch panel at 200 %, a window snapped to half the
    // screen) tighten the wordmark and the cards instead of clipping them.
    readonly property bool compact: page.height < 600

    // Scrolls rather than overflowing: centred, the menu ran off both edges of
    // any window shorter than itself — the app's own minimum size included.
    Flickable {
        id: flick

        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            bottom: banner.visible ? banner.top : parent.bottom
            bottomMargin: banner.visible ? 8 : 0
        }
        contentWidth: width
        contentHeight: menu.implicitHeight + 2 * menu.y
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar {
            policy: flick.contentHeight > flick.height ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
        }

        ColumnLayout {
            id: menu

            x: (flick.width - width) / 2
            // Centred while it fits, top-aligned with a margin once it does not.
            y: Math.max(page.compact ? 12 : 24, (flick.height - implicitHeight) / 2)
            width: Math.min(560, flick.width - 48)
            spacing: page.compact ? 6 : 10

            Text {
                text: "stenograf"
                color: Theme.text
                font.pixelSize: page.compact ? Theme.textHeading + 6 : Theme.textDisplay
                font.weight: Font.Light
                font.letterSpacing: 1.5
                Layout.alignment: Qt.AlignHCenter
            }

            Text {
                text: "local meeting transcription"
                color: Theme.muted
                font.pixelSize: Theme.textLabel
                Layout.alignment: Qt.AlignHCenter
                Layout.bottomMargin: page.compact ? 6 : 18
            }

            Repeater {
                model: page.app.menu

                delegate: Button {
                    id: card

                    required property var modelData
                    required property int index

                    Layout.fillWidth: true
                    // Sized by its two lines, so a larger text scale grows the
                    // card instead of pushing the lines into its border.
                    leftPadding: 18
                    rightPadding: 18
                    topPadding: page.compact ? 9 : 13
                    bottomPadding: page.compact ? 9 : 13
                    focus: card.index === 0
                    // Tabbing to a card below the fold brings it into view.
                    onActiveFocusChanged: if (activeFocus) {
                        var top = card.mapToItem(menu, 0, 0).y + menu.y;
                        if (top < flick.contentY)
                            flick.contentY = Math.max(0, top - 12);
                        else if (top + card.height > flick.contentY + flick.height)
                            flick.contentY = Math.min(flick.contentHeight - flick.height, top + card.height - flick.height + 12);
                    }

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
                            font.pixelSize: Theme.textReading
                            font.weight: Font.Medium
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }

                        Text {
                            text: card.modelData.description
                            color: Theme.muted
                            font.pixelSize: Theme.textSmall
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }
                }
            }
        }
    }

    // One line about updates, at the foot of the menu and only there: no
    // dialog, no notification, and nothing at all while a meeting runs
    // (`quiet`). It is the whole update interface besides the menu-bar entry.
    Rectangle {
        id: banner

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
                    font.pixelSize: Theme.textBody
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }

                // Selectable: on an install that cannot update itself this is
                // the command to paste into a terminal.
                TextEdit {
                    visible: text !== ""
                    text: page.updates.state.detail
                    color: Theme.muted
                    font.pixelSize: Theme.textSmall
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

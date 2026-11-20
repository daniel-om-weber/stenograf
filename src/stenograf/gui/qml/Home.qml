// The menu: the meeting first and largest, the other workflows as smaller
// tiles, the way out last and quiet. Each entry carries a line about what it
// does — the audience is people who don't know the subcommands. Every entry is
// a real Button, so the whole menu is keyboard-reachable (tab + space).
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page

    property var app
    // The update line's object (gui/updates.py). Not a page, but fetched like
    // one, so this file needs no second injected property.
    readonly property var updates: page.app.screen("Update")
    readonly property var entries: page.app.menu
    readonly property var main: page.entries[0]
    readonly property var others: page.entries.slice(1, page.entries.length - 1)
    readonly property var leave: page.entries[page.entries.length - 1]

    function choose(entry) {
        // quit_app(), not Qt.quit(): leaving with a meeting running has to stop
        // capture and let the finalize land first.
        if (entry.page === "quit")
            page.app.quit_app();
        else
            page.app.open(entry.page);
    }

    // Tabbing to an entry below the fold brings it into view.
    function reveal(item) {
        var top = item.mapToItem(flick.contentItem, 0, 0).y;
        if (top < flick.contentY)
            flick.contentY = Math.max(0, top - 12);
        else if (top + item.height > flick.contentY + flick.height)
            flick.contentY = Math.min(flick.contentHeight - flick.height, top + item.height - flick.height + 12);
    }

    // Short windows (a 7-inch panel at 200 %, a window snapped to half the
    // screen) tighten the header and the entries instead of clipping them.
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
        contentHeight: menu.implicitHeight + 2 * menu.margin
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar {
            policy: flick.contentHeight > flick.height ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
        }

        ColumnLayout {
            id: menu

            readonly property int margin: page.compact ? 12 : 24

            x: (flick.width - width) / 2
            // Centred while it fits, top-aligned with a margin once it does not.
            y: Math.max(menu.margin, (flick.height - implicitHeight) / 2)
            width: Math.min(600, flick.width - 48)
            spacing: page.compact ? 8 : 12

            // The header: the app's own icon over its name, so the window
            // and the Dock tile visibly belong together.
            Image {
                source: page.app.icon_url
                visible: !page.compact
                sourceSize.width: 128
                sourceSize.height: 128
                Layout.preferredWidth: 64
                Layout.preferredHeight: 64
                Layout.alignment: Qt.AlignHCenter
                smooth: true
                mipmap: true
            }

            Text {
                text: "stenograf"
                color: Theme.text
                font.pixelSize: page.compact ? Theme.textTitle : Theme.textDisplay
                font.weight: Font.Light
                font.letterSpacing: 1.5
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: page.compact ? 0 : -4
            }

            Text {
                text: "local meeting transcription"
                color: Theme.muted
                font.pixelSize: Theme.textLabel
                Layout.alignment: Qt.AlignHCenter
                Layout.bottomMargin: page.compact ? 4 : 14
            }

            // The meeting: what the app is for, so it is the one entry in the
            // accent colour, and the one that has focus when the window opens.
            Button {
                id: hero

                focus: true
                leftPadding: 16
                rightPadding: 16
                topPadding: page.compact ? 10 : 16
                bottomPadding: page.compact ? 10 : 16
                Layout.fillWidth: true
                Accessible.name: page.main.label
                onClicked: page.choose(page.main)
                onActiveFocusChanged: if (activeFocus)
                    page.reveal(hero)

                background: Rectangle {
                    radius: 14
                    color: hero.hovered ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.2) : Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.12)
                    border.width: 1
                    border.color: hero.hovered || hero.activeFocus ? Theme.accent : Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.45)

                    Behavior on color {
                        ColorAnimation {
                            duration: 90
                        }
                    }
                }

                contentItem: RowLayout {
                    spacing: 14

                    Rectangle {
                        Layout.preferredWidth: page.compact ? 36 : 44
                        Layout.preferredHeight: page.compact ? 36 : 44
                        radius: width / 2
                        color: Theme.accent

                        Icon {
                            anchors.centerIn: parent
                            name: page.main.icon
                            color: Theme.accentText
                            size: page.compact ? 18 : 22
                        }
                    }

                    ColumnLayout {
                        spacing: 2
                        Layout.fillWidth: true

                        Text {
                            text: page.main.label
                            color: Theme.text
                            font.pixelSize: Theme.textTitle
                            font.weight: Font.DemiBold
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }

                        Text {
                            text: page.main.description
                            color: Theme.muted
                            font.pixelSize: Theme.textBody
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }

                    Icon {
                        name: "chevron-right"
                        color: hero.hovered ? Theme.accent : Theme.muted
                        size: 20
                    }
                }
            }

            // The other workflows, two to a row where they fit.
            GridLayout {
                columns: menu.width >= 480 ? 2 : 1
                columnSpacing: page.compact ? 8 : 12
                rowSpacing: page.compact ? 8 : 12
                Layout.fillWidth: true

                Repeater {
                    model: page.others

                    delegate: Button {
                        id: tile

                        required property var modelData

                        leftPadding: 14
                        rightPadding: 14
                        topPadding: page.compact ? 9 : 13
                        bottomPadding: page.compact ? 9 : 13
                        Layout.fillWidth: true
                        // Equal columns, whatever the length of each label.
                        Layout.preferredWidth: 1
                        Layout.fillHeight: true
                        Accessible.name: tile.modelData.label
                        onClicked: page.choose(tile.modelData)
                        onActiveFocusChanged: if (activeFocus)
                            page.reveal(tile)

                        background: Rectangle {
                            radius: 12
                            color: tile.hovered ? Theme.surfaceHi : Theme.surface
                            border.width: 1
                            border.color: tile.hovered || tile.activeFocus ? Theme.accent : Theme.line

                            Behavior on color {
                                ColorAnimation {
                                    duration: 90
                                }
                            }
                        }

                        contentItem: RowLayout {
                            spacing: 12

                            Icon {
                                name: tile.modelData.icon
                                color: tile.hovered || tile.activeFocus ? Theme.accent : Theme.muted
                                size: 20
                                Layout.alignment: Qt.AlignTop
                                Layout.topMargin: 1
                            }

                            ColumnLayout {
                                spacing: 3
                                Layout.fillWidth: true
                                // Labels on one line across a row, however
                                // many lines each description takes.
                                Layout.alignment: Qt.AlignTop

                                Text {
                                    text: tile.modelData.label
                                    color: Theme.text
                                    font.pixelSize: Theme.textReading
                                    font.weight: Font.Medium
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }

                                Text {
                                    text: tile.modelData.description
                                    color: Theme.muted
                                    font.pixelSize: Theme.textSmall
                                    wrapMode: Text.WordWrap
                                    maximumLineCount: 2
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }
                }
            }

            Btn {
                id: quit

                text: page.leave.label
                glyph: page.leave.icon
                flat: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: page.compact ? 0 : 4
                onClicked: page.choose(page.leave)
                onActiveFocusChanged: if (activeFocus)
                    page.reveal(quit)
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

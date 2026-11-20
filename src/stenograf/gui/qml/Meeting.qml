// The live meeting: header (phase, elapsed, language, profile), the committed
// captions, the dim per-channel tail of what is still provisional, and a footer
// with the status line, the meeting folder and the one button that matters.
//
// Captions arrive as signals and are appended to a ListModel — not through the
// state map, which would re-evaluate every binding on the screen for each new
// line. Nothing here animates: a pulsing REC dot would hold the compositor
// awake for the entire meeting, against a pipeline tuned to ~0.6 W.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page

    property var app
    property var screen

    readonly property string phase: page.screen.state.phase
    // One reading column for the committed captions and the live line under
    // them, centred: left-anchored, a maximized window left its right half
    // blank. The speaker column is shared so the names line up across both.
    readonly property int readingWidth: Math.min(page.width - 36 - 14, 900)
    readonly property int whoWidth: 110

    // A caption is plain text, but the live line is drawn as styled text (to
    // colour its provisional tail inline), so its markup characters must not
    // be read as tags.
    function escaped(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    // The transcript line a "Who spoke?" quote was clicked for, shown in the
    // accent colour until the next click; -1 = none.
    property int jumpedLine: -1

    function jumpTo(line) {
        page.jumpedLine = line;
        captions.positionViewAtIndex(line, ListView.Center);
    }
    readonly property color phaseColor: page.phase === "rec" ? Theme.rec : page.phase === "done" ? Theme.good : page.phase === "failed" ? Theme.bad : Theme.busy

    Component.onCompleted: if (page.screen)
        page.screen.opened()

    // Escape does what the button does: stop while capturing, leave once there
    // is nothing left to stop. Never a plain "go back" — popping this page
    // would leave a meeting running with no way to reach its Stop. Off while
    // the finish card is open: Escape in one of its fields reads as "cancel
    // this edit", and leaving would drop everything typed into the card.
    Shortcut {
        sequences: [StandardKey.Cancel]
        enabled: !names.visible
        onActivated: page.screen.stop()
    }

    ListModel {
        id: captionModel
    }

    Connections {
        target: page.screen

        function onCommitted(channel, who, line) {
            captionModel.append({
                "channel": channel,
                "who": who,
                "stamp": "",
                "line": line,
                "faded": false
            });
            captions.positionViewAtEnd();
        }

        function onCleared() {
            captionModel.clear();
        }

        function onRestored(entries) {
            // The finalize swap: diarized speakers replace the channel-coarse
            // live captions wholesale.
            captionModel.clear();
            page.jumpedLine = -1;
            for (var i = 0; i < entries.length; ++i) {
                captionModel.append({
                    "channel": "",
                    "who": entries[i].speaker,
                    "stamp": entries[i].time,
                    "line": entries[i].text,
                    "faded": entries[i].provisional
                });
            }
            captions.positionViewAtEnd();
        }
    }

    Rectangle {
        id: header

        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
        }
        height: 56
        color: Theme.surface

        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: Theme.line
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 22
            anchors.rightMargin: 22
            spacing: 12

            Rectangle {
                width: 9
                height: 9
                radius: 4.5
                color: page.phaseColor
                Layout.alignment: Qt.AlignVCenter
            }

            Text {
                text: page.screen.state.phaseLabel
                color: Theme.text
                font.pixelSize: Theme.textBody
                font.weight: Font.DemiBold
                font.letterSpacing: 0.6
            }

            Text {
                text: page.screen.state.elapsed
                color: Theme.text
                font.pixelSize: Theme.textBody
                font.family: Theme.mono
            }

            // Also the row's spring: it takes the spare width and gives it back
            // first when the channel chips need room.
            Text {
                // The language appears once detected — at the finalize, or at
                // the start when a meeting type fixes it.
                text: (page.screen.state.language ? page.screen.state.language + "  ·  " : "") + page.screen.state.profile
                color: Theme.muted
                font.pixelSize: Theme.textSmall
                elide: Text.ElideRight
                horizontalAlignment: Text.AlignRight
                Layout.fillWidth: true
            }

            Repeater {
                model: page.phase === "rec" ? page.screen.state.channels : []

                ChannelChip {
                    required property string modelData

                    channel: modelData
                    muted: modelData === "mic" ? page.screen.state.micMuted : page.screen.state.systemMuted
                    enabled: page.screen.state.canStop
                    onClicked: page.screen.mute(modelData, !muted)
                }
            }
        }
    }

    // The finish card, over the transcript once it is on disk; its quotes
    // jump to their lines below, so a few of those lines stay in sight however
    // short the window — at a flat 70 % of it the transcript vanished at
    // 960x500 (2026-11-20). The card scrolls inside itself, Done pinned. The
    // run's notes wait for it.
    FinishCard {
        id: names

        screen: page.screen
        visible: page.phase === "done" && page.screen.state.namesOpen
        anchors {
            top: header.bottom
            left: parent.left
            right: parent.right
        }
        height: visible ? Math.min(implicitHeight, Math.max(180, page.height - header.height - footer.height - interim.height - 110)) : 0
        onJump: line => page.jumpTo(line)
    }

    ListView {
        id: captions

        anchors {
            top: names.visible ? names.bottom : header.bottom
            left: parent.left
            right: parent.right
            bottom: interim.top
        }
        anchors.margins: 18
        clip: true
        spacing: 14
        model: captionModel
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar {}

        // A full-width row holding the centred one: a vertical ListView
        // places its delegates at x 0, so a delegate cannot centre itself.
        delegate: Item {
            id: row

            required property int index
            required property string channel
            required property string who
            required property string stamp
            required property string line
            required property bool faded

            readonly property bool jumped: row.index === page.jumpedLine

            width: captions.width - 14
            height: rowLayout.implicitHeight

            RowLayout {
                id: rowLayout

                x: Math.max(0, (row.width - width) / 2)
                width: page.readingWidth
                spacing: 14

                Text {
                    text: row.who
                    color: row.jumped ? Theme.accent : row.channel === "mic" ? Theme.mic : row.channel === "system" ? Theme.remote : Theme.text
                    font.pixelSize: Theme.textBody
                    font.weight: Font.DemiBold
                    horizontalAlignment: Text.AlignRight
                    elide: Text.ElideRight
                    // Wide enough for a name once the voices are named, not only
                    // for "Remote-1".
                    Layout.preferredWidth: page.whoWidth
                    Layout.alignment: Qt.AlignTop
                }

                Text {
                    text: row.stamp
                    visible: row.stamp.length > 0
                    color: Theme.dim
                    font.pixelSize: Theme.textSmall
                    font.family: Theme.mono
                    Layout.alignment: Qt.AlignTop
                }

                Text {
                    text: row.line
                    color: row.jumped ? Theme.accent : row.faded ? Theme.muted : Theme.text
                    font.pixelSize: Theme.textReading
                    lineHeight: 1.35
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
    }

    // What is still in flight: the open committed line (bright) and the grey
    // provisional tail behind it, per channel.
    ColumnLayout {
        id: interim

        anchors {
            left: parent.left
            bottom: footer.top
            leftMargin: 18 + Math.max(0, (page.width - 36 - 14 - page.readingWidth) / 2)
            bottomMargin: 8
        }
        width: page.readingWidth
        spacing: 6

        Repeater {
            model: page.screen.state.tails

            delegate: RowLayout {
                id: tail

                required property var modelData

                spacing: 14
                Layout.fillWidth: true

                Text {
                    text: tail.modelData.speaker
                    color: tail.modelData.channel === "mic" ? Theme.mic : Theme.remote
                    opacity: 0.5
                    font.pixelSize: Theme.textBody
                    font.weight: Font.DemiBold
                    horizontalAlignment: Text.AlignRight
                    Layout.preferredWidth: page.whoWidth
                    Layout.alignment: Qt.AlignTop
                }

                // The open line and its grey provisional tail as one paragraph,
                // so the tail wraps on from where the line ends.
                Text {
                    text: page.escaped(tail.modelData.open) + (tail.modelData.open && tail.modelData.tail ? " " : "") + "<font color=\"" + Theme.dim + "\">" + page.escaped(tail.modelData.tail) + "</font>"
                    textFormat: Text.StyledText
                    lineHeight: 1.35
                    color: Theme.text
                    font.pixelSize: Theme.textReading
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                }
            }
        }
    }

    Rectangle {
        id: footer

        anchors {
            left: parent.left
            right: parent.right
            bottom: parent.bottom
        }
        height: 64
        color: Theme.surface

        Rectangle {
            width: parent.width
            height: 1
            color: Theme.line
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 22
            anchors.rightMargin: 18
            spacing: 16

            ColumnLayout {
                spacing: 2
                Layout.fillWidth: true

                Text {
                    text: page.screen.state.status
                    visible: text.length > 0
                    color: page.phase === "failed" ? Theme.bad : Theme.muted
                    font.pixelSize: Theme.textSmall
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }

                PathLink {
                    text: page.screen.state.folder
                    visible: text.length > 0
                    Layout.fillWidth: true
                    onActivated: page.screen.reveal()
                }
            }

            Btn {
                id: copyButton

                // Confirmed on the button itself: the status line beside it is
                // the notes step's, and would be overwritten within a second.
                property bool copied: false

                text: copied ? "Copied" : "Copy transcript"
                glyph: copied ? "check" : "copy"
                visible: page.phase === "done" && page.screen.state.transcriptReady && !page.screen.state.namesOpen
                implicitWidth: 150
                onClicked: copied = page.screen.copy()

                Timer {
                    interval: 1500
                    running: copyButton.copied
                    onTriggered: copyButton.copied = false
                }
            }

            Btn {
                text: "Edit"
                glyph: "pencil"
                visible: page.phase === "done" && page.screen.state.transcriptReady && !page.screen.state.namesOpen
                implicitWidth: 100
                onClicked: page.screen.open_names()
            }

            Btn {
                // Before capture is up (canStop false) the button cancels the
                // starting run instead of stopping a live one — screen.stop()
                // routes both, so it is enabled through the whole "rec" phase.
                text: page.phase === "rec" ? (page.screen.state.canStop ? "Stop & finalize" : "Cancel") : page.phase === "finalizing" ? "Finalizing…" : "Back to menu"
                glyph: page.phase === "rec" && page.screen.state.canStop ? "stop" : ""
                primary: page.phase === "rec"
                enabled: page.phase !== "finalizing"
                implicitWidth: 150
                onClicked: page.screen.stop()
            }
        }
    }
}

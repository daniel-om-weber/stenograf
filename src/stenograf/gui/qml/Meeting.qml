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
    readonly property color phaseColor: page.phase === "rec" ? Theme.rec : page.phase === "done" ? Theme.good : page.phase === "failed" ? Theme.bad : Theme.busy

    Component.onCompleted: if (page.screen)
        page.screen.opened()

    // Escape does what the button does: stop while capturing, leave once there
    // is nothing left to stop. Never a plain "go back" — popping this page
    // would leave a meeting running with no way to reach its Stop.
    Shortcut {
        sequences: [StandardKey.Cancel]
        onActivated: page.screen.stop()
    }

    ListModel {
        id: captionModel
    }

    Connections {
        target: page.screen

        function onCommitted(who, line) {
            captionModel.append({
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
            for (var i = 0; i < entries.length; ++i) {
                captionModel.append({
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
                font.pixelSize: 13
                font.weight: Font.DemiBold
                font.letterSpacing: 0.6
            }

            Text {
                text: page.screen.state.elapsed
                color: Theme.text
                font.pixelSize: 13
                font.family: Theme.mono
            }

            Item {
                Layout.fillWidth: true
            }

            Text {
                text: page.screen.state.language + "  ·  " + page.screen.state.profile
                color: Theme.muted
                font.pixelSize: 12
            }
        }
    }

    ListView {
        id: captions

        anchors {
            top: header.bottom
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

        delegate: RowLayout {
            id: row

            required property string who
            required property string stamp
            required property string line
            required property bool faded

            width: Math.min(captions.width - 14, 900)
            spacing: 14

            Text {
                text: row.who
                color: row.who === "You" ? Theme.mic : row.who === "Remote" ? Theme.remote : Theme.text
                font.pixelSize: 13
                font.weight: Font.DemiBold
                horizontalAlignment: Text.AlignRight
                Layout.preferredWidth: 78
                Layout.alignment: Qt.AlignTop
            }

            Text {
                text: row.stamp
                visible: row.stamp.length > 0
                color: Theme.dim
                font.pixelSize: 12
                font.family: Theme.mono
                Layout.alignment: Qt.AlignTop
            }

            Text {
                text: row.line
                color: row.faded ? Theme.muted : Theme.text
                font.pixelSize: 15
                lineHeight: 1.35
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }

    // What is still in flight: the open committed line (bright) and the grey
    // provisional tail behind it, per channel.
    ColumnLayout {
        id: interim

        anchors {
            left: parent.left
            right: parent.right
            bottom: footer.top
            leftMargin: 18
            rightMargin: 18
            bottomMargin: 8
        }
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
                    color: tail.modelData.speaker === "You" ? Theme.mic : Theme.remote
                    opacity: 0.5
                    font.pixelSize: 13
                    font.weight: Font.DemiBold
                    horizontalAlignment: Text.AlignRight
                    Layout.preferredWidth: 78
                    Layout.alignment: Qt.AlignTop
                }

                Text {
                    text: tail.modelData.open
                    visible: text.length > 0
                    color: Theme.text
                    font.pixelSize: 15
                    wrapMode: Text.WordWrap
                    Layout.maximumWidth: 460
                    Layout.alignment: Qt.AlignTop
                }

                Text {
                    text: tail.modelData.tail
                    visible: text.length > 0
                    color: Theme.dim
                    font.pixelSize: 15
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
                    font.pixelSize: 12
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }

                Text {
                    text: page.screen.state.folder
                    visible: text.length > 0
                    color: Theme.dim
                    font.pixelSize: 11
                    elide: Text.ElideMiddle
                    Layout.fillWidth: true
                }
            }

            Btn {
                text: page.phase === "rec" ? "Stop & finalize" : page.phase === "finalizing" ? "Finalizing…" : "Back to menu"
                primary: page.phase === "rec"
                enabled: page.phase === "finalizing" ? false : page.phase !== "rec" || page.screen.state.canStop
                implicitWidth: 150
                onClicked: page.screen.stop()
            }
        }
    }
}

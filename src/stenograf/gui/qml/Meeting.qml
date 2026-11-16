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

    // "Who spoke?": one row per nameable speaker once the transcript is on
    // disk. Saving names the voices for good (later meetings recognise them);
    // the run's notes wait until this is saved or skipped.
    Rectangle {
        id: names

        visible: page.phase === "done" && page.screen.state.namesOpen
        anchors {
            top: header.bottom
            left: parent.left
            right: parent.right
        }
        height: visible ? Math.min(namesColumn.implicitHeight + 36, page.height * 0.6) : 0
        color: Theme.surface

        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: Theme.line
        }

        Flickable {
            anchors.fill: parent
            anchors.margins: 18
            contentHeight: namesColumn.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

            ColumnLayout {
                id: namesColumn

                width: Math.min(parent.width - 14, 900)
                spacing: 10

                Text {
                    text: "Who spoke?"
                    color: Theme.text
                    font.pixelSize: 15
                    font.weight: Font.DemiBold
                }

                Text {
                    text: "Name a voice once and later meetings recognise it." + (page.screen.state.knownNames.length > 0 ? "  Known: " + page.screen.state.knownNames : "")
                    color: Theme.muted
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }

                Repeater {
                    id: speakerRows

                    model: page.screen.state.speakers

                    delegate: ColumnLayout {
                        id: speakerRow

                        required property var modelData
                        // A recognised speaker shows the name; "Not …?" opens
                        // the field to correct it. A refused answer reopens it.
                        property bool editing: !speakerRow.modelData.recognised || speakerRow.modelData.refused.length > 0
                        readonly property string answer: nameField.text.trim()

                        spacing: 4
                        Layout.fillWidth: true

                        Rectangle {
                            height: 1
                            color: Theme.line
                            Layout.fillWidth: true
                        }

                        RowLayout {
                            spacing: 12
                            Layout.fillWidth: true
                            Layout.topMargin: 6

                            ColumnLayout {
                                spacing: 2
                                Layout.fillWidth: true

                                RowLayout {
                                    spacing: 8

                                    Text {
                                        text: speakerRow.modelData.label
                                        color: Theme.text
                                        font.pixelSize: 14
                                        font.weight: Font.DemiBold
                                    }

                                    Text {
                                        text: speakerRow.modelData.recognised ? "recognised" : ""
                                        visible: text.length > 0
                                        color: Theme.good
                                        font.pixelSize: 11
                                    }

                                    Text {
                                        text: speakerRow.modelData.spoke
                                        color: Theme.dim
                                        font.pixelSize: 11
                                    }
                                }

                                // Each quote jumps to its line in the transcript below,
                                // where the lines around it often say who this is.
                                Repeater {
                                    model: speakerRow.modelData.quotes

                                    delegate: Text {
                                        id: quote

                                        required property var modelData

                                        text: quote.modelData.text
                                        color: quoteMouse.containsMouse ? Theme.text : Theme.muted
                                        font.pixelSize: 12
                                        wrapMode: Text.WordWrap
                                        Layout.fillWidth: true

                                        MouseArea {
                                            id: quoteMouse

                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: page.jumpTo(quote.modelData.line)
                                        }
                                    }
                                }
                            }

                            Btn {
                                readonly property bool playing: page.screen.state.playing === speakerRow.modelData.label
                                readonly property int next: (page.screen.state.clipNext[speakerRow.modelData.label] || 0) % Math.max(1, speakerRow.modelData.clips)

                                visible: speakerRow.modelData.clips > 0
                                text: playing ? "■ Stop" : "▶ Voice " + (next + 1) + "/" + speakerRow.modelData.clips
                                implicitWidth: 110
                                onClicked: page.screen.play(speakerRow.modelData.label)
                            }

                            Field {
                                id: nameField

                                visible: speakerRow.editing
                                text: speakerRow.modelData.name
                                placeholderText: "Name"
                                enabled: !page.screen.state.namesBusy
                                Layout.preferredWidth: 190
                                onAccepted: saveButton.clicked()
                                onTextEdited: page.screen.hold_names()
                            }

                            Btn {
                                visible: !speakerRow.editing
                                text: "Not " + speakerRow.modelData.label + "?"
                                implicitWidth: 190
                                onClicked: {
                                    page.screen.hold_names();
                                    speakerRow.editing = true;
                                    nameField.forceActiveFocus();
                                }
                            }
                        }

                        RowLayout {
                            visible: speakerRow.modelData.refused.length > 0
                            spacing: 12
                            Layout.fillWidth: true

                            Text {
                                text: speakerRow.modelData.refused
                                color: Theme.busy
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }

                            Btn {
                                text: "Add anyway"
                                enabled: !page.screen.state.namesBusy && speakerRow.answer.length > 0
                                implicitWidth: 190
                                onClicked: {
                                    var one = {};
                                    one[speakerRow.modelData.label] = speakerRow.answer;
                                    page.screen.save_names(one, [speakerRow.modelData.label]);
                                }
                            }
                        }
                    }
                }

                RowLayout {
                    spacing: 10
                    Layout.fillWidth: true
                    Layout.topMargin: 6

                    Text {
                        // The auto-close countdown ([speakers] names_panel_timeout_s);
                        // typing stops it.
                        text: page.screen.state.namesBusy ? "Saving…" : page.screen.state.namesCountdown > 0 ? "Closes in " + page.screen.state.namesCountdown + " s — typing keeps it open" : ""
                        color: Theme.muted
                        font.pixelSize: 12
                        Layout.fillWidth: true
                    }

                    Btn {
                        text: "Skip"
                        enabled: !page.screen.state.namesBusy
                        onClicked: page.screen.skip_names()
                    }

                    Btn {
                        id: saveButton

                        text: "Save names"
                        primary: true
                        enabled: !page.screen.state.namesBusy
                        onClicked: {
                            var answers = {};
                            for (var i = 0; i < speakerRows.count; ++i) {
                                var row = speakerRows.itemAt(i);
                                if (row && row.editing && row.answer.length > 0)
                                    answers[row.modelData.label] = row.answer;
                            }
                            page.screen.save_names(answers, []);
                        }
                    }
                }
            }
        }
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

        delegate: RowLayout {
            id: row

            required property int index
            required property string channel
            required property string who
            required property string stamp
            required property string line
            required property bool faded

            readonly property bool jumped: row.index === page.jumpedLine

            width: Math.min(captions.width - 14, 900)
            spacing: 14

            Text {
                text: row.who
                color: row.jumped ? Theme.accent : row.channel === "mic" ? Theme.mic : row.channel === "system" ? Theme.remote : Theme.text
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
                color: row.jumped ? Theme.accent : row.faded ? Theme.muted : Theme.text
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
                    color: tail.modelData.channel === "mic" ? Theme.mic : Theme.remote
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
                visible: page.phase === "done" && page.screen.state.transcriptReady
                implicitWidth: 150
                onClicked: copied = page.screen.copy()

                Timer {
                    interval: 1500
                    running: copyButton.copied
                    onTriggered: copyButton.copied = false
                }
            }

            Btn {
                text: "Name speakers"
                visible: page.phase === "done" && page.screen.state.speakers.length > 0 && !page.screen.state.namesOpen
                implicitWidth: 150
                onClicked: page.screen.open_names()
            }

            Btn {
                // Before capture is up (canStop false) the button cancels the
                // starting run instead of stopping a live one — screen.stop()
                // routes both, so it is enabled through the whole "rec" phase.
                text: page.phase === "rec" ? (page.screen.state.canStop ? "Stop & finalize" : "Cancel") : page.phase === "finalizing" ? "Finalizing…" : "Back to menu"
                primary: page.phase === "rec"
                enabled: page.phase !== "finalizing"
                implicitWidth: 150
                onClicked: page.screen.stop()
            }
        }
    }
}

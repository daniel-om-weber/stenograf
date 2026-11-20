// The finish card: what a meeting needs once it is over — a title (which names
// its folder), who spoke, whether and how to write notes, and the files —
// in one card with icons where headings would be. Done applies it all and the
// notes start; the run waits for it (or for the timeout, or for the screen
// being left, which keep the choices as the start form made them).
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Rectangle {
    id: card

    property var screen
    // `state` would shadow Item.state, the QML states string.
    readonly property var st: card.screen.state
    readonly property bool busy: card.st.namesBusy

    // Tracks the state map, so it fires on the card's opening only.
    readonly property bool opening: card.st.namesOpen

    signal jump(int line)

    // The inputs are filled here, once per opening, and never bound to the
    // state map: that map is one property, so every status line or countdown
    // tick re-evaluates every binding on it — and a binding re-evaluated over
    // a field the user has typed into writes the old value back.
    function load() {
        titleField.text = card.st.title;
        notesSwitch.checked = card.st.notesOn;
        presetCombo.model = card.st.presets;
        presetCombo.currentIndex = Math.max(0, card.st.presets.findIndex(p => p.value === card.st.preset));
        contextField.text = card.st.context;
    }

    onOpeningChanged: if (card.opening)
        card.load()
    Component.onCompleted: card.load()

    function done() {
        var answers = {};
        for (var i = 0; i < speakerRows.count; ++i) {
            var row = speakerRows.itemAt(i);
            if (row && row.editing && row.answer.length > 0 && row.answer !== row.modelData.label)
                answers[row.modelData.label] = row.answer;
        }
        card.screen.finish(answers, [], titleField.text, notesSwitch.checked, presetCombo.value === undefined ? "" : presetCombo.value, contextField.text);
    }

    color: Theme.surface
    implicitHeight: column.implicitHeight + column.spacing + actions.implicitHeight + 36

    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width
        height: 1
        color: Theme.line
    }

    // Only the title, the voices and the notes scroll; the action row stays
    // pinned below them. The card gets only part of the window, so a Done at
    // the end of the scrolled column fell out of sight from five voices up in
    // the default window (measured 2026-11-16).
    Flickable {
        id: flick

        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            bottom: actions.top
            margins: 18
            bottomMargin: column.spacing
        }
        contentHeight: column.implicitHeight
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        // Basic's scrollbar shows only while scrolling, which makes a clipped
        // card look complete.
        ScrollBar.vertical: ScrollBar {
            policy: flick.contentHeight > flick.height ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
        }

        ColumnLayout {
            id: column

            x: Math.max(0, (parent.width - width) / 2)
            width: Math.min(parent.width - 14, 720)
            spacing: 14

            // Title, and the folder name it becomes.
            ColumnLayout {
                spacing: 4
                Layout.fillWidth: true

                TextField {
                    id: titleField

                    placeholderText: "Untitled meeting"
                    font.pixelSize: Theme.textHeading
                    color: Theme.text
                    placeholderTextColor: Theme.dim
                    selectionColor: Theme.accent
                    selectedTextColor: Theme.accentText
                    leftPadding: 8
                    rightPadding: 34
                    enabled: !card.busy
                    Layout.fillWidth: true
                    Accessible.name: "Meeting title"
                    onTextEdited: card.screen.hold_names()

                    background: Rectangle {
                        radius: 8
                        color: titleField.activeFocus || titleField.hovered ? Theme.surfaceHi : "transparent"
                        border.width: titleField.activeFocus ? 1 : 0
                        border.color: Theme.accent
                    }

                    Icon {
                        anchors.right: parent.right
                        anchors.rightMargin: 10
                        anchors.verticalCenter: parent.verticalCenter
                        name: "pencil"
                        color: Theme.dim
                        size: 15
                    }
                }

                RowLayout {
                    spacing: 6
                    Layout.leftMargin: 8

                    Icon {
                        name: "folder"
                        color: Theme.dim
                        size: 12
                    }

                    Text {
                        // The folder too: the page is built before the meeting
                        // exists, and an unchanged title would never re-ask.
                        text: card.st.folder ? card.screen.preview(titleField.text) : ""
                        color: Theme.dim
                        font.pixelSize: Theme.textSmall
                        font.family: Theme.mono
                        elide: Text.ElideMiddle
                        Layout.fillWidth: true
                    }
                }
            }

            Rectangle {
                visible: speakerRows.count > 0
                height: 1
                color: Theme.line
                Layout.fillWidth: true
            }

            // Who spoke: one row per nameable voice.
            Repeater {
                id: speakerRows

                model: card.st.speakers

                delegate: ColumnLayout {
                    id: speakerRow

                    required property var modelData
                    // An unknown voice opens with its field; a recognised one shows
                    // its name as a button, because a wrong match must be one click
                    // from fixing. A refused answer reopens the field.
                    property bool editing: !speakerRow.modelData.recognised || speakerRow.modelData.refused.length > 0
                    readonly property string answer: nameField.text.trim()
                    readonly property string initials: speakerRow.modelData.recognised ? speakerRow.modelData.label.split(/\s+/).map(w => w.charAt(0)).join("").slice(0, 2).toUpperCase() : ""

                    function edit() {
                        card.screen.hold_names();
                        speakerRow.editing = true;
                        nameField.text = speakerRow.modelData.recognised ? speakerRow.modelData.label : speakerRow.modelData.name;
                        nameField.forceActiveFocus();
                        nameField.selectAll();
                    }

                    spacing: 6
                    Layout.fillWidth: true

                    RowLayout {
                        spacing: 12
                        Layout.fillWidth: true

                        Rectangle {
                            width: 32
                            height: 32
                            radius: 16
                            color: speakerRow.modelData.recognised ? Qt.darker(Theme.accent, 3.2) : Theme.surfaceHi
                            border.width: speakerRow.modelData.recognised ? 0 : 1
                            border.color: Theme.line
                            Layout.alignment: Qt.AlignTop

                            Text {
                                anchors.centerIn: parent
                                visible: speakerRow.initials.length > 0
                                text: speakerRow.initials
                                color: Theme.accent
                                font.pixelSize: Theme.textSmall
                                font.weight: Font.DemiBold
                            }

                            Icon {
                                anchors.centerIn: parent
                                visible: speakerRow.initials.length === 0
                                name: "user"
                                color: Theme.muted
                                size: 16
                            }
                        }

                        ColumnLayout {
                            spacing: 3
                            Layout.fillWidth: true

                            // The recognised name: a button, with the pencil saying so.
                            AbstractButton {
                                id: nameButton

                                visible: !speakerRow.editing
                                hoverEnabled: true
                                implicitHeight: 28
                                implicitWidth: nameRow.implicitWidth + 16
                                Accessible.role: Accessible.Button
                                Accessible.name: "Change the name " + speakerRow.modelData.label
                                ToolTip.visible: hovered
                                ToolTip.delay: 500
                                ToolTip.text: "Not " + speakerRow.modelData.label + "? Click to change"
                                onClicked: speakerRow.edit()

                                background: Rectangle {
                                    radius: 8
                                    color: nameButton.hovered ? Theme.control : "transparent"
                                    border.width: 1
                                    border.color: nameButton.hovered ? Theme.dim : Theme.line
                                }

                                contentItem: Item {
                                    Row {
                                        id: nameRow

                                        anchors.centerIn: parent
                                        spacing: 7

                                        Text {
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: speakerRow.modelData.label
                                            color: Theme.text
                                            font.pixelSize: Theme.textLabel
                                        }

                                        Icon {
                                            anchors.verticalCenter: parent.verticalCenter
                                            name: "check-circle"
                                            color: Theme.good
                                            size: 14
                                        }

                                        Icon {
                                            anchors.verticalCenter: parent.verticalCenter
                                            name: "pencil"
                                            color: nameButton.hovered ? Theme.text : Theme.muted
                                            size: 13
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                visible: speakerRow.editing
                                spacing: 6

                                Field {
                                    id: nameField

                                    text: speakerRow.modelData.name
                                    placeholderText: "Who is this?"
                                    implicitHeight: 32
                                    enabled: !card.busy
                                    Layout.preferredWidth: 220
                                    ToolTip.visible: activeFocus && card.st.knownNames.length > 0 && text.length === 0
                                    ToolTip.text: "Known: " + card.st.knownNames
                                    onAccepted: card.done()
                                    onTextEdited: card.screen.hold_names()
                                }

                                IconButton {
                                    // Back to the recognised name, unchanged.
                                    visible: speakerRow.modelData.recognised && speakerRow.modelData.refused.length === 0
                                    glyph: "x"
                                    tip: "Keep " + speakerRow.modelData.label
                                    implicitWidth: 32
                                    implicitHeight: 32
                                    onClicked: speakerRow.editing = false
                                }
                            }

                            // The longest thing they said; a click shows it in the
                            // transcript below, where the lines around it help.
                            Text {
                                id: quote

                                readonly property var first: speakerRow.modelData.quotes.length > 0 ? speakerRow.modelData.quotes[0] : null

                                visible: first !== null
                                text: first ? first.text : ""
                                color: quoteMouse.containsMouse ? Theme.text : Theme.muted
                                font.pixelSize: Theme.textSmall
                                font.italic: true
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                                Layout.leftMargin: 2

                                MouseArea {
                                    id: quoteMouse

                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: card.jump(quote.first.line)
                                }
                            }
                        }

                        RowLayout {
                            spacing: 4
                            Layout.alignment: Qt.AlignTop
                            Layout.topMargin: 6

                            Icon {
                                name: "clock"
                                color: Theme.dim
                                size: 12
                            }

                            Text {
                                text: speakerRow.modelData.spoke
                                color: Theme.dim
                                font.pixelSize: Theme.textSmall
                                font.family: Theme.mono
                            }
                        }

                        IconButton {
                            readonly property bool playing: card.st.playing === speakerRow.modelData.label

                            visible: speakerRow.modelData.clips > 0
                            round: true
                            glyph: playing ? "stop" : "play"
                            tip: playing ? "Stop" : "Hear this voice"
                            Layout.alignment: Qt.AlignTop
                            onClicked: card.screen.play(speakerRow.modelData.label)
                        }
                    }

                    RowLayout {
                        visible: speakerRow.modelData.refused.length > 0
                        spacing: 12
                        Layout.fillWidth: true
                        Layout.leftMargin: 44

                        Text {
                            text: speakerRow.modelData.refused
                            color: Theme.busy
                            font.pixelSize: Theme.textSmall
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }

                        Btn {
                            text: "Add anyway"
                            enabled: !card.busy && speakerRow.answer.length > 0
                            onClicked: {
                                var one = {};
                                one[speakerRow.modelData.label] = speakerRow.answer;
                                card.screen.save_names(one, [speakerRow.modelData.label]);
                            }
                        }
                    }
                }
            }

            Rectangle {
                visible: !card.st.notesLocked
                height: 1
                color: Theme.line
                Layout.fillWidth: true
            }

            // Notes: on or off, which meeting type's setup, and a line of context.
            RowLayout {
                visible: !card.st.notesLocked
                spacing: 12
                Layout.fillWidth: true

                Icon {
                    name: "sparkles"
                    color: notesSwitch.checked ? Theme.accent : Theme.dim
                    size: 18
                }

                Toggle {
                    id: notesSwitch

                    label: "Notes"
                    Layout.fillWidth: false
                    onToggled: card.screen.hold_names()
                }

                Combo {
                    id: presetCombo

                    visible: card.st.presets.length > 1
                    enabled: notesSwitch.checked && !card.busy
                    implicitHeight: 32
                    Layout.preferredWidth: 170
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: "Meeting type — which notes setup to use"
                    onActivated: card.screen.hold_names()
                }

                Field {
                    id: contextField

                    placeholderText: "What was it about? (optional)"
                    enabled: notesSwitch.checked && !card.busy
                    implicitHeight: 32
                    leftPadding: 32
                    Layout.fillWidth: true
                    onTextEdited: card.screen.hold_names()

                    Icon {
                        anchors.left: parent.left
                        anchors.leftMargin: 10
                        anchors.verticalCenter: parent.verticalCenter
                        name: "message"
                        color: Theme.dim
                        size: 14
                    }
                }
            }
        }
    }

    // The files, and Done, lined up with the column above.
    RowLayout {
        id: actions

        x: flick.x + column.x
        width: column.width
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 18
        spacing: 8

        IconButton {
            glyph: "folder"
            tip: "Open the folder"
            onClicked: card.screen.reveal()
        }

        IconButton {
            glyph: "file-text"
            tip: "Open the transcript"
            onClicked: card.screen.open_transcript()
        }

        IconButton {
            id: copyButton

            property bool copied: false

            glyph: copied ? "check" : "copy"
            tip: copied ? "Copied" : "Copy the transcript"
            onClicked: copied = card.screen.copy()

            Timer {
                interval: 1500
                running: copyButton.copied
                onTriggered: copyButton.copied = false
            }
        }

        IconButton {
            id: discardButton

            // Two clicks: the first arms it, the second moves the folder
            // to the trash. Disarms itself after a few seconds.
            danger: true
            glyph: "trash"
            tip: armed ? "Click again to move this meeting to the trash" : "Discard this meeting"
            enabled: !card.busy
            onClicked: {
                card.screen.hold_names();  // the countdown must not close the card between the clicks
                if (armed)
                    card.screen.discard();
                else
                    armed = true;
            }

            Timer {
                interval: 4000
                running: discardButton.armed
                onTriggered: discardButton.armed = false
            }
        }

        Item {
            Layout.fillWidth: true
        }

        RowLayout {
            // The auto-close countdown; typing stops it.
            visible: card.st.namesCountdown > 0
            spacing: 4

            Icon {
                name: "clock"
                color: Theme.dim
                size: 12
            }

            Text {
                text: card.st.namesCountdown + " s"
                color: Theme.dim
                font.pixelSize: Theme.textSmall
                font.family: Theme.mono
            }
        }

        Btn {
            text: card.busy ? "Saving…" : "Done"
            glyph: card.busy ? "" : "check"
            primary: true
            enabled: !card.busy
            implicitWidth: 110
            onClicked: card.done()
        }
    }
}

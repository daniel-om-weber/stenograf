// The effective configuration, read-only, with every value's provenance. Not a
// form: editing happens in settings.toml itself, which the Open button hands to
// whatever this desktop uses for .toml files.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Panel {
    id: page

    property var screen
    readonly property var updates: page.app.screen("Update")

    heading: "Settings"
    hint: "Every value in effect and where it comes from. To change one, edit settings.toml."
    cardWidth: 760

    // Name, value and tag side by side where they fit, stacked where not.
    readonly property bool wide: page.width >= 720

    Component.onCompleted: if (page.screen)
        page.screen.opened()

    // Not an editor: picking a meeting type re-renders the same read-only report
    // with that [meetings.*] overlay applied, which is the only way to see what
    // a type actually changes. Hidden when no presets are defined.
    RowLayout {
        spacing: 10
        visible: page.screen.state.presets.length > 1
        Layout.fillWidth: true
        Layout.bottomMargin: 4

        Text {
            text: "Meeting type"
            color: Theme.text
            font.pixelSize: Theme.textBody
        }

        Combo {
            id: preset

            model: page.screen.state.presets
            // Driven by the screen's own selection, so Reload (which re-reads
            // the file and may drop a deleted preset) and the report can never
            // disagree about which type is on display.
            currentIndex: Math.max(0, preset.model.findIndex(entry => entry.value === page.screen.state.preset))
            Layout.fillWidth: true
            onActivated: page.screen.show(preset.value || "")
        }
    }

    // Where the file is: selectable, so the path can be pasted into a terminal.
    RowLayout {
        spacing: 8
        Layout.fillWidth: true

        Icon {
            name: "file-text"
            color: Theme.muted
            size: 14
            Layout.alignment: Qt.AlignTop
            Layout.topMargin: 1
        }

        TextEdit {
            text: page.screen.state.path + (page.screen.state.present ? "" : "  (not present — every value is a default)")
            color: Theme.muted
            font.pixelSize: Theme.textSmall
            font.family: Theme.mono
            wrapMode: Text.WrapAnywhere
            readOnly: true
            selectByMouse: true
            selectionColor: Theme.accent
            selectedTextColor: Theme.accentText
            Layout.fillWidth: true
        }
    }

    Repeater {
        model: page.screen.state.notes

        Text {
            required property string modelData

            text: modelData
            color: Theme.remote
            font.pixelSize: Theme.textBody
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    // A file that does not parse, or one nothing on this desktop opens.
    TextEdit {
        visible: text.length > 0
        text: page.screen.state.text
        color: Theme.bad
        font.pixelSize: Theme.textSmall
        font.family: Theme.mono
        wrapMode: Text.Wrap
        readOnly: true
        selectByMouse: true
        selectionColor: Theme.accent
        selectedTextColor: Theme.accentText
        Layout.fillWidth: true
    }

    // One group per table, one row per key: the name, the value as it would
    // be written in the file (selectable, to paste it there), and a tag for
    // where it came from. Values that are not defaults are what this screen is
    // usually opened to find, so only those tags are coloured.
    Repeater {
        model: page.screen.state.sections

        delegate: ColumnLayout {
            id: section

            required property var modelData

            spacing: 4
            Layout.fillWidth: true

            SectionLabel {
                text: section.modelData.table
                font.capitalization: Font.MixedCase
                font.family: Theme.mono
            }

            Repeater {
                model: section.modelData.rows

                delegate: GridLayout {
                    id: entry

                    required property var modelData

                    // Long values (an attendee list, a path) start folded to
                    // their first line, so one key cannot bury the rest.
                    readonly property bool lengthy: entry.modelData.value.length > 90
                    property bool unfolded: false
                    readonly property string source: entry.modelData.source

                    columns: page.wide ? 3 : 1
                    columnSpacing: 12
                    rowSpacing: 2
                    Layout.fillWidth: true
                    Layout.topMargin: page.wide ? 0 : 4

                    Text {
                        text: entry.modelData.key
                        color: Theme.text
                        font.pixelSize: Theme.textBody
                        wrapMode: Text.WrapAnywhere
                        Layout.preferredWidth: page.wide ? 220 : -1
                        Layout.fillWidth: !page.wide
                        Layout.alignment: Qt.AlignTop
                    }

                    ColumnLayout {
                        spacing: 0
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop

                        TextEdit {
                            text: entry.lengthy && !entry.unfolded ? entry.modelData.value.slice(0, 80) + "…" : entry.modelData.value
                            color: entry.source === "default" ? Theme.muted : Theme.text
                            font.pixelSize: Theme.textSmall
                            font.family: Theme.mono
                            wrapMode: Text.Wrap
                            readOnly: true
                            selectByMouse: true
                            selectionColor: Theme.accent
                            selectedTextColor: Theme.accentText
                            Layout.fillWidth: true
                            Layout.topMargin: 1
                        }

                        Text {
                            visible: entry.lengthy
                            text: entry.unfolded ? "Show less" : "Show all"
                            color: fold.containsMouse ? Theme.accent : Theme.dim
                            font.pixelSize: Theme.textSmall

                            MouseArea {
                                id: fold

                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: entry.unfolded = !entry.unfolded
                            }
                        }
                    }

                    Rectangle {
                        readonly property color tint: entry.source === "default" ? Theme.dim : entry.source === "settings.toml" ? Theme.accent : entry.source.charAt(0) === "$" ? Theme.busy : Theme.remote

                        implicitWidth: tag.implicitWidth + 14
                        implicitHeight: tag.implicitHeight + 4
                        radius: height / 2
                        color: entry.source === "default" ? "transparent" : Qt.rgba(tint.r, tint.g, tint.b, 0.14)
                        border.width: entry.source === "default" ? 1 : 0
                        border.color: Theme.line
                        Layout.alignment: Qt.AlignTop | (page.wide ? Qt.AlignRight : Qt.AlignLeft)
                        Layout.maximumWidth: page.wide ? 220 : -1

                        Text {
                            id: tag

                            anchors.centerIn: parent
                            width: Math.min(implicitWidth, parent.Layout.maximumWidth > 0 ? parent.Layout.maximumWidth - 14 : implicitWidth)
                            text: entry.source
                            color: entry.source === "default" ? Theme.dim : parent.tint
                            font.pixelSize: Theme.textSmall
                            elide: Text.ElideRight
                        }
                    }
                }
            }
        }
    }

    // The one place to ask on demand; the daily check needs no button.
    RowLayout {
        spacing: 10
        Layout.fillWidth: true
        Layout.topMargin: 10

        Text {
            text: page.updates.state.summary
            color: Theme.muted
            font.pixelSize: Theme.textSmall
            elide: Text.ElideRight
            Layout.fillWidth: true
        }

        Btn {
            text: "Check for updates"
            onClicked: page.updates.check_now()
        }
    }

    actions: [
        Btn {
            text: "Open settings.toml"
            Layout.fillWidth: true
            onClicked: page.screen.edit()
        },

        Btn {
            text: "Reload"
            Layout.fillWidth: true
            onClicked: page.screen.opened()
        },

        Btn {
            text: "Back"
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    ]
}

// The readiness report. Same vocabulary as `steno doctor` — passed, an
// optional check the machine can healthily fail, a real problem — drawn as
// icons in a fixed column, problems first (the screen sorts them), with the
// verdict above the list rather than under it.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Panel {
    id: page

    property var screen
    // Name and detail side by side where they fit, stacked where not.
    readonly property bool wide: page.width >= 680
    readonly property int problems: page.screen.state.checks.filter(c => c.state === "bad").length

    heading: "Check setup"
    hint: "Models, permissions, helpers and audio devices."
    cardWidth: 720
    busy: page.screen.state.busy

    Component.onCompleted: if (page.screen)
        page.screen.opened()

    // The verdict.
    Rectangle {
        visible: page.screen.state.status.length > 0
        implicitHeight: verdict.implicitHeight + 16
        radius: 8
        color: page.screen.state.busy ? Theme.surfaceHi : page.problems > 0 ? Qt.rgba(Theme.bad.r, Theme.bad.g, Theme.bad.b, 0.12) : Qt.rgba(Theme.good.r, Theme.good.g, Theme.good.b, 0.10)
        Layout.fillWidth: true
        Layout.bottomMargin: 6

        RowLayout {
            id: verdict

            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            spacing: 10

            Icon {
                visible: !page.screen.state.busy
                name: page.problems > 0 ? "alert-circle" : "check-circle"
                color: page.problems > 0 ? Theme.bad : Theme.good
                size: 18
            }

            Text {
                text: page.screen.state.status
                color: Theme.text
                font.pixelSize: Theme.textBody
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }

    Repeater {
        model: page.screen.state.checks

        delegate: RowLayout {
            id: check

            required property var modelData

            spacing: 10
            Layout.fillWidth: true

            Icon {
                name: check.modelData.state === "good" ? "check-circle" : check.modelData.state === "optional" ? "minus-circle" : "alert-circle"
                color: check.modelData.state === "good" ? Theme.good : check.modelData.state === "optional" ? Theme.busy : Theme.bad
                size: 16
                Layout.alignment: Qt.AlignTop
                Layout.topMargin: 1
            }

            GridLayout {
                columns: page.wide ? 2 : 1
                columnSpacing: 12
                rowSpacing: 1
                Layout.fillWidth: true

                Text {
                    text: check.modelData.name
                    color: Theme.text
                    font.pixelSize: Theme.textBody
                    font.weight: Font.Medium
                    wrapMode: Text.WordWrap
                    Layout.preferredWidth: page.wide ? 190 : -1
                    Layout.fillWidth: !page.wide
                    Layout.alignment: Qt.AlignTop
                }

                Text {
                    text: check.modelData.detail
                    color: check.modelData.state === "bad" ? Theme.text : Theme.muted
                    font.pixelSize: Theme.textBody
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
    }

    actions: [
        Btn {
            text: "Check again"
            enabled: !page.screen.state.busy
            Layout.fillWidth: true
            onClicked: page.screen.opened()
        },

        Btn {
            text: "Back"
            enabled: !page.screen.state.busy
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    ]
}

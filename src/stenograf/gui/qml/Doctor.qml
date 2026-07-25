// The readiness report. Same vocabulary as `steno doctor`: ✓ passed, ○ an
// optional check the machine can healthily fail, ✗ a real problem.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Panel {
    id: page

    property var screen

    heading: "Check setup"
    hint: "Models, permissions, helpers and audio devices."
    cardWidth: 720
    busy: page.screen.state.busy

    Component.onCompleted: if (page.screen)
        page.screen.opened()

    Repeater {
        model: page.screen.state.checks

        delegate: RowLayout {
            id: check

            required property var modelData

            spacing: 10
            Layout.fillWidth: true

            Text {
                text: check.modelData.state === "good" ? "✓" : check.modelData.state === "optional" ? "○" : "✗"
                color: check.modelData.state === "good" ? Theme.good : check.modelData.state === "optional" ? Theme.busy : Theme.bad
                font.pixelSize: 13
                Layout.alignment: Qt.AlignTop
            }

            Text {
                text: check.modelData.name
                color: Theme.text
                font.pixelSize: 13
                font.weight: Font.Medium
                wrapMode: Text.WordWrap
                Layout.preferredWidth: 175
                Layout.alignment: Qt.AlignTop
            }

            Text {
                text: check.modelData.detail
                color: Theme.muted
                font.pixelSize: 13
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }

    Text {
        text: page.screen.state.status
        visible: text.length > 0
        color: Theme.muted
        font.pixelSize: 13
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
        Layout.topMargin: 8
    }

    RowLayout {
        Layout.fillWidth: true
        Layout.topMargin: 8

        Btn {
            text: "Back"
            enabled: !page.screen.state.busy
            Layout.fillWidth: true
            onClicked: page.app.back()
        }
    }
}

// One recorded channel in the meeting header: what it is, whether it is being
// transcribed, and the switch that mutes it. One control, so the state and the
// way to change it are never read apart. A live chip takes its channel's caption
// colour, tying it to the "You"/"Remote" lines below; a muted one turns red with
// the struck-through icon video calls use for the same thing.
import QtQuick
import QtQuick.Controls.Basic

AbstractButton {
    id: chip

    property string channel
    property bool muted: false
    readonly property bool mic: channel === "mic"
    readonly property color ink: muted ? Theme.muteInk : Theme.text

    implicitHeight: 30
    implicitWidth: row.implicitWidth + leftPadding + rightPadding
    leftPadding: 10
    rightPadding: 12
    hoverEnabled: true
    opacity: enabled ? 1 : 0.5

    Accessible.role: Accessible.CheckBox
    Accessible.checkable: true
    Accessible.checked: muted
    Accessible.name: (mic ? "Mute microphone" : "Mute system audio")

    ToolTip.visible: hovered
    ToolTip.delay: 600
    ToolTip.text: mic ? (muted ? "Unmute your microphone" : "Mute your microphone — nothing you say is transcribed") : (muted ? "Unmute system audio" : "Mute system audio — remote voices are not transcribed")

    background: Rectangle {
        radius: height / 2
        color: chip.muted ? Theme.muteFill : chip.hovered ? Theme.control : Theme.surfaceHi
        border.width: 1
        border.color: chip.muted ? Theme.rec : chip.hovered ? Theme.dim : Theme.line

        Behavior on color {
            ColorAnimation {
                duration: 90
            }
        }
    }

    contentItem: Item {
        Row {
            id: row

            anchors.centerIn: parent
            spacing: 6

            Icon {
                anchors.verticalCenter: parent.verticalCenter
                name: chip.mic ? (chip.muted ? "mic-off" : "mic") : (chip.muted ? "volume-off" : "volume")
                color: chip.muted ? Theme.muteInk : chip.mic ? Theme.mic : Theme.remote
                size: 15
            }

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: (chip.mic ? "You" : "Remote") + (chip.muted ? " · muted" : " · live")
                color: chip.ink
                font.pixelSize: Theme.textSmall
            }
        }
    }
}

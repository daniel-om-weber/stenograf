// A drop-down over a [{label, value}] list from Python. The Basic style draws
// nothing on its own, which is the point: every part is ours.
import QtQuick
import QtQuick.Controls.Basic

ComboBox {
    id: control

    // The selected entry and its value, read straight out of the model rather
    // than through ComboBox's role resolution: the model is a plain list of
    // {label, value} maps from Python, and indexing it is one less thing that
    // can quietly resolve to undefined.
    readonly property var currentOption: control.model && control.currentIndex >= 0 ? control.model[control.currentIndex] : null
    readonly property var value: control.currentOption ? control.currentOption[control.valueRole] : undefined

    implicitHeight: 36
    font.pixelSize: 13
    textRole: "label"
    valueRole: "value"

    background: Rectangle {
        radius: 8
        color: Theme.surfaceHi
        border.width: 1
        border.color: control.activeFocus ? Theme.accent : Theme.line
    }

    contentItem: Text {
        leftPadding: 12
        rightPadding: 28
        text: control.currentOption ? control.currentOption[control.textRole] : ""
        font: control.font
        color: Theme.text
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Text {
        x: control.width - width - 12
        y: (control.height - height) / 2
        text: "▾"
        color: Theme.muted
        font.pixelSize: 12
    }

    delegate: ItemDelegate {
        id: option

        required property var modelData
        required property int index

        width: control.width
        implicitHeight: 32
        highlighted: control.highlightedIndex === option.index

        background: Rectangle {
            color: option.highlighted ? Theme.surfaceHi : "transparent"
        }

        contentItem: Text {
            leftPadding: 12
            text: option.modelData[control.textRole]
            color: Theme.text
            font.pixelSize: 13
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    popup: Popup {
        y: control.height + 4
        width: control.width
        implicitHeight: Math.min(list.contentHeight + 2, 260)
        padding: 1

        background: Rectangle {
            radius: 8
            color: Theme.surface
            border.width: 1
            border.color: Theme.line
        }

        contentItem: ListView {
            id: list

            clip: true
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}
        }
    }
}

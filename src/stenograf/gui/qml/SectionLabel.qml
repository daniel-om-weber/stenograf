// A quiet heading over a group of controls in a form: enough to say which
// switches belong together, without competing with the page's own heading.
import QtQuick
import QtQuick.Layouts

Text {
    color: Theme.muted
    font.pixelSize: Theme.textSmall
    font.weight: Font.DemiBold
    font.letterSpacing: 0.8
    font.capitalization: Font.AllUppercase
    Layout.fillWidth: true
    Layout.topMargin: 4
}

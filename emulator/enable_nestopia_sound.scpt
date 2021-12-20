tell application "System Events"
    tell process "Nestopia"
        set frontmost to true
        set desiredCheckboxStatus to true as boolean
        click menu item "Preferences..." of menu "Nestopia" of menu bar 1
        set theCheckbox to checkbox "Enable Sound" of window 1
        tell theCheckbox
            set checkboxStatus to value of theCheckbox as boolean
            if checkboxStatus is not desiredCheckboxStatus then click theCheckbox
        end tell
        click button 1 of window 1
    end tell
end tell

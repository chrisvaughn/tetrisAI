tell application "System Events"
	tell process "Nestopia"
		set frontmost to true
		set desiredCheckboxStatus to true as boolean

		click menu "Nestopia" of menu bar 1
		delay 1 -- Wait for the menu to appear

		click menu item "Settings…" of menu "Nestopia" of menu bar 1
		delay 1 -- Wait for the menu to appear

		set theCheckbox to checkbox "Enable Sound" of window 1
		tell theCheckbox
			set checkboxStatus to value as boolean
			if checkboxStatus is not desiredCheckboxStatus then click
		end tell
		click button 1 of window 1
	end tell
end tell
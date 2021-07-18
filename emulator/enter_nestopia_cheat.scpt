tell application "System Events"
	tell process "Nestopia"
		set frontmost to true
		click menu item "Enter Cheat Code..." of menu "Options" of menu bar 1
		tell text field 1 of window 1
			click
			set value to "ZAYAKP"
			key code 36
			key code 36
		end tell
	end tell
end tell

#| Slang `if' test. |#

main {
	if true {
		stdio.println(1)
	} elif false {
		stdio.println(-1)
	} elif true {
		stdio.println(-2)
	} else {
		stdio.println(-3)
	}

	if false {
		stdio.println(-1)
	} elif true {
		stdio.println(1)
	} else {
		stdio.println(-2)
	}

	if false {
		stdio.println(-1)
	} elif false {
		stdio.println(-2)
	} else {
		stdio.println(1)
	}

	if false {
		stdio.println(-1)
	} else {
		stdio.println(1)
		stdio.println("OK!")
	}
}

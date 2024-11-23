#| Slang `do' test. |#

main {
	do: stdio.println(1/0)
	catch Exception ex {
		stdio.println(1)
		raise
	} catch Exception {
		stdio.println(2)
		raise
	} catch: stdio.println(3); #resume
	else: stdio.println(-1)
	finally {
		stdio.println(0)
	}
}

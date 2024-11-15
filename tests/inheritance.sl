#| Slang `inheritance' test. |#

class A {
	int x = 3
}

class B(A) {
	int y = 5
}

main {
	A a
	B b

	stdio.println(a.x)
	stdio.println(b.x, b.y)
}

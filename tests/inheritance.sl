#| Slang `inheritance' test. |#

class A {
	int x = 3

	int f(int x) = (x + 3)
}

class B {
	int y = 5
}

class C < B < A {
	int f(int x) = (((super.f)(7)) + x)
}

main {
	A a
	C c

	stdio.println(a.x, (a.f)(1))
	stdio.println(c.x, c.y, (c.f)(1))
}

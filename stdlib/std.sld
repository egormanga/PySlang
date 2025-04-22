# `i8`, `i16`, `i32`, `i64`, `i128` — fixed-size integer
# `u8`, `u16`, `u32`, `u64`, `u128` — fixed-size unsigned integer
# `f8`, `f16`, `f32`, `f64`, `f128` — fixed-size IEEE-754 floating point number
# `uf8`, `uf16`, `uf32`, `uf64`, `uf128` — fixed-size unsigned floating point number
# `c8`, `c16`, `c32`, `c64`, `c128` — fixed-size complex number
# `uc8`, `uc16`, `uc32`, `uc64`, `uc128` — fixed-size unsigned complex number
# `int` — unsized («big») integer
# `uint` — unsized unsigned integer
# `float` — unsized floating point number
# `ufloat` — unsized unsigned floating point
# `complex` — unsized complex number
# `ucomplex` — unsized unsigned complex number
# `frac` — unsized 2-adic number [https://www.cs.utoronto.ca/~hehner/NR.pdf]
# `bool` — binary logical (boolean) value
# `byte` — single byte
# `char` — UTF-8 character
# `str` — UTF-8 string
# `list` — typed list
# `tuple` — typed tuple
# `map` — untyped mapping (dictionary)
# `set` — untyped set
# `void` — nothing
# `auto` — compile-time type deduction based on value


class void {}

class bool {
	castable to int;

	bool !_;
}

class int {
	castable to bool;
	castable to float;

	int +_;
	int -_;
	int ~_;
	int ++_;
	int --_;

	int _!;
	int _++;
	int _--;
	int _**;

	int _ +int;
	int _ -int;
	int _ *int;
	int _ //int;
	int _ %int;
	int _ **int;
	int _ <<int;
	int _ >>int;
	int _ &int;
	int _ |int;
	int _ ^int;

	range _ to int;

	int popcount();

	int tobase(int base);

	int length(int base=2);
}

class float {
	float +_;
	float -_;
	bool !_;

	float _ +float;
	float _ -float;
	float _ *float;
	float _ /float;
	int _ //float;
	float _ %float;
	float _ **float;

	int round();
	int round(int prec);

	bool iswhole();
}

class char {
	castable to str;

	bool !_;

	char _ +char;
	char _ +int;
	char _ -char;
	char _ -int;
	char _ *int;

	range _ to char;
}

class str {
	bool !_;

	iterable char;

	char _[int];

	str _ +str;
	str _ *int;

	int length();

	str strip(str chars);
}

class range {
	typename type;

	iterable type;

	type _[int];
}

class tuple {
	typename types[];

	iterable types;

	bool !_;

	types[0] _[int];
}

class list {
	typename type;

	iterable type;

	bool !_;

	list _ +list;
	list _ *int;

	type _[int];

	void append(type item);

	void insert(int index, type item);

	type pop();
	type pop(int index);

	void reverse();

	void sort();

	0.rettype each(func);
}

class func {
	typename argtypes[];
	typename rettype;

	rettype _(*argtypes);

	rettype map(iterable);
}

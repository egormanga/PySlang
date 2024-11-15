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

	bool operator !;
}

class int {
	castable to bool;
	castable to float;

	int +operator;
	int -operator;
	int ~operator;
	int ++operator;
	int --operator;

	int operator!;
	int operator++;
	int operator--;
	int operator**;

	int operator +int;
	int operator -int;
	int operator *int;
	int operator //int;
	int operator %int;
	int operator **int;
	int operator <<int;
	int operator >>int;
	int operator &int;
	int operator |int;
	int operator ^int;

	range operator 'to' int;

	int popcount();

	int tobase(int base);

	int length(int base=2);
}

class float {
	float +operator;
	float -operator;
	bool !operator;

	float operator +float;
	float operator -float;
	float operator *float;
	float operator /float;
	int operator //float;
	float operator %float;
	float operator **float;

	int round();
	int round(int prec);

	bool iswhole();
}

class char {
	castable to str;

	bool !operator;

	char operator +char;
	char operator +int;
	char operator -char;
	char operator -int;
	char operator *int;

	range operator 'to' char;
}

class str {
	bool !operator;

	iterable char;

	char operator [int];

	str operator +str;
	str operator *int;

	int length();

	str strip(str chars);
}

class range {
	typename type;

	iterable type;

	type operator [int];
}

class tuple {
	typename types[];

	iterable types;

	bool !operator;

	types[0] operator [int];
}

class list {
	typename type;

	iterable type;

	bool !operator;

	list operator +list;
	list operator *int;

	type operator [int];

	void append(type item);

	void insert(int index, type item);

	type pop();
	type pop(int index);

	void reverse();

	void sort();

	0.rettype each(function);
}

class function {
	typename argtypes[];
	typename rettype;

	rettype operator (*argtypes);

	rettype map(iterable);
}

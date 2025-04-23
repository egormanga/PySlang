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

class range {
	typename type;

	iterable type;

	#type _[int];
}

class int {
	castable to bool;
	castable to float;

	int +_;   # absolute value
	int -_;
	int ~_;
	int ++_;  # pre-increment
	int --_;  # pre-decrement

	int _!;   # factorial
	int _++;  # post-increment
	int _--;  # post-decrement
	int _**;  # square

	int _ +int;
	int _ -int;
	int _ *int;
	int _ //int;  # integer division
	int _ %int;
	int _ **int;  # power
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
	float _ ^/float;    # ceiling division
	float _ ~/float;    # rounding division
	float _ //float;    # flooring division
	float _ %float;
	float _ **float;  # power

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

	str _ +str;  # concatenation
	str _ *int;  # repetition

	int length();

	str strip(str chars);
}

class tuple {
	typename types[];

	iterable types;

	bool !_;

	types[$1] _[int];
}

class list {
	typename type;

	iterable type;

	bool !_;

	list _ +list;  # concatenation
	list _ *int;   # repetition

	type _[int];

	void append(type item);

	void insert(int index, type item);

	type pop();
	type pop(int index);

	void reverse();

	void sort();

	#list of $1.rettype each(func);
}

class func {
	typename rettype;
	typename argtypes[];

	rettype _(*argtypes);

	rettype map(iterable);
}

import sld:std:*;
import sld:int:*;

class fd {
	str read();
	str read(int n);

	int write(str data);
}

class stdio {
	str readln();

	void print(*args);

	void println(*args);
}

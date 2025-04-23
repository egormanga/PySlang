#include <stdio.h>

extern void module(void) asm("\"<module>\"");
static void __attribute__((constructor)) _init_module(void) { module(); }

void __attribute__((weak)) main() {}

void stdio_println(int *) asm("stdio.println");
void stdio_println(int *x) {
	printf("%d\n", *x);
}

# Referência didática, System V ABI x86-64 Linux, sintaxe AT&T.
# Não executada pelo protótipo. double sum(double x, double y).
.text
.globl sum
.type sum, @function
sum:
    addsd %xmm1, %xmm0
    ret
.size sum, .-sum
.section .note.GNU-stack,"",@progbits

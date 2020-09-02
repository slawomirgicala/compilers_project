from ex3 import calc

examples = [
    'a = 1',
    'xd = sin cos 1/2*4+3-5; xd',
    'xd = sin cos (1/2*4+3-5); xd',
    'a = sin (-143 + 12 ^ 2); a; a + 1; 1<2+1; if(1<2){2-1}',
    'a = sin 0;a; if(1>2){2-1}else{69}; a=1; while(a < 10){a = a+1}; a; for(i = 0;i<5;i = i+1){i}',
    'int a = 1; a = 2; str i = "hello"; i'
]


if __name__ == "__main__":
    for i, example in enumerate(examples, 1):
        calc.draw_ast(example, 'ast/example_' + str(i) + '.png')

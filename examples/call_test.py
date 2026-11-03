class Person:
    def __call__(self,name):
        print('hello'+'__call__'+name)

    def hello(self,name):
        print('hello'+'hello'+name)

person = Person()       #必须需要有一个对象
person('zansi')
person.hello('lisi')
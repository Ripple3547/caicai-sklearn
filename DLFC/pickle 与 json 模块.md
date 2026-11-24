1. 序列化与反序列化是为了解决内存中对象的持久化与传输问题；
2. Python中提供了pickle和json两个模块进行序列化与反序列化；
3. dumps()和dump()用于序列化，loads()和load()用于反序列化；
4. pickle模块能序列化任何对象，序列化结果为bites类型，只适合于Python机器之间交互；
5. json模块只能序列化Python基本类型，序列化结果为json格式字符串，适合不同开发语言之间交互。
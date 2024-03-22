#按文本字符串分析
# https://github.com/Cyberbolt/Cemotion
from cemotion import Cemotion

str_text1 = '配置顶级，不解释，手机需要的各个方面都很完美'
str_text2 = '院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！'

c = Cemotion()
print('"', str_text1 , '"\n' , '预测值:{:6f}'.format(c.predict(str_text1) ) , '\n')
print('"', str_text2 , '"\n' , '预测值:{:6f}'.format(c.predict(str_text2) ) , '\n')

list_text = [
    '内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！',
    '总而言之，是一家不会再去的店。',
    '湖大真好看，桃子湖真漂亮',
    '你好，初次见面，请多指教'
]

for each in c.predict(list_text):
    print(each)

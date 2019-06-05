import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from sklearn.metrics import mean_squared_error
import os

os.chdir('D:\\Projeler\\BMMU-proje\\dizin') # csv dosyamızın olduğu dizin

xim = pd.read_csv('dataset.csv', usecols=['dolar']) # verileri sutun sutun çekiyoruz
yim = pd.read_csv('dataset.csv', usecols=['export'])

x=xim.iloc[:,0].values # Çekilen veriler başka bir formatta olduğu için kullanacağımız dizi formatına dönüştürüyoruz.
y=yim.iloc[:,0].values


def gradient_descent(x,y):
    m_degeri = b_degeri = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.06


    for i in range(iterations):
        y_predicted = m_degeri * x + b_degeri  # Regresyon modelimiz    ===     y = mx+b
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)]) # Mean Squared Error
        c= mean_squared_error(y, y_predicted) # Mean Squared Error (Bu da aynı şeyi yapıyor)
        md = -(2/n)*sum(x*(y-y_predicted))  # Mean Squared Error fonksiyonunun kısmi türevi
        bd = -(2/n)*sum(y-y_predicted)  # Mean Squared Error fonksiyonunun kısmi türevi
        m_degeri = m_degeri - learning_rate * md   # Gredient descent fonkiyonu
        b_degeri = b_degeri - learning_rate * bd   # Gredient descent fonkiyonu
        print ("m {}, b {}, cost {} , cost2  {}  , iteration {} ".format(m_degeri,b_degeri,cost ,c , i))

    # Ekranda Gredient descent algoritmasının gösterilmesi
    plt.scatter(x, y, color='red')
    plt.title('Dolar kuruna Göre ihracat Tahmini Regresyon Modeli')
    plt.xlabel('Dolar kuru')
    plt.ylabel('İhracat Miktarı(milyar $)')
    plt.grid(True)
    plt.plot(x,y_predicted, color='blue')
    plt.show()

gradient_descent(x,y)

# lineer regresyon algoritması,  Gredient descent algoritmasından çıkan sonucun kontrolü için gerekli
def regresyon(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean((xs * ys))) / (mean(xs)**2 - mean(xs**2))
    b = mean(ys) - (m * mean(xs))
    return m, b

m, b = regresyon(x, y)
line = [(m * x) + b for x in range(0,8)]  # regresyon fonksiyonundan gelen degerlerle doğru oluşturuluyor

print('Lineer regresyonla oluşan deger')
print('m {}, b {}'.format(m,b))

plt.plot(line, color='blue')  # doğrunun çizilmesi
plt.scatter(x, y, color = 'red')  # datasetteki verilerin çizilmesi
plt.title('Dolar kuruna Göre ihracat Tahmini Regresyon Modeli')
plt.xlabel('Dolar kuru')
plt.ylabel('İhracat Miktarı(milyar $)')
plt.grid(True)
plt.show()  # çizilecek verilerin ekranda gösterilmesi
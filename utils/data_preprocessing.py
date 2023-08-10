import preprocessor as p
import pandas as pd
import re


CSV_DATA_PATH = ""
SAVE_PATH = ""


def preprocessing(x):
    p.set_options(p.OPT.URL, p.OPT.NUMBER) # removing urls
    x=p.tokenize(x)
    p.set_options(p.OPT.RESERVED, p.OPT.MENTION) # removing twitter reserved words and mentions
    x=p.clean(x)
    x=re.sub(r":\)+", "radość", x) # replacement of emoticons with appropriate words
    x=re.sub(r":-\)+", "radość", x)
    x=re.sub(r":\(+", "smutek", x)
    x=re.sub(r":-\(+", "smutek", x)
    x=re.sub(r":C+", "duży smutek", x)
    x=re.sub(r":-C+", "duży smutek", x)
    x=re.sub(r":\(+", "płacz", x)
    x=re.sub(r":-\(+", "płacz", x)
    x=re.sub(r";\(+", "płacz", x)
    x=re.sub(r";-\(+", "płacz", x)
    x=re.sub(r":p+", "śmiech", x)
    x=re.sub(r":-p+", "śmiech", x)
    x=re.sub(r":P+", "śmiech", x)
    x=re.sub(r":-P+", "śmiech", x)
    x=re.sub(r":D+", "szeroki uśmiech", x)
    x=re.sub(r":-D+", "szeroki uśmiech", x)
    x=re.sub(r":\*+", "pocałunek", x)
    x=re.sub(r":-\*+", "pocałunek", x)
    x=re.sub(r":O+", "zdziwienie", x)
    x=re.sub(r":-O+", "zdziwienie", x)
    x=re.sub(r"xD+", "śmiech", x)
    x=re.sub(r"XD+", "śmiech", x)
    x=re.sub(r"Xd+", "śmiech", x)
    x=re.sub(r"xd+", "śmiech", x)
    x=re.sub(r":\/+", "grymas", x)
    x=re.sub(r":-\/+", "grymas", x)
    x=re.sub(r":\|+", "niezdecydowanie", x)
    x=re.sub(r":-\|+", "niezdecydowanie", x)
    x=re.sub(r";\)+", "uśmiech z przymrużeniem oka", x)
    x=re.sub(r";-\)+", "uśmiech z przymrużeniem oka", x)
    x=re.sub(r":'+\(+", "smutek", x)
    x=re.sub(r":'+\(+", "smutek", x)
    x=re.sub(r' +', ' ', x) # removing multiple spaces inside the sentence
    x=x.lstrip()
    x=x.rstrip()
    return x


def main():
    data = pd.read_csv(CSV_DATA_PATH, header=None)
    data.dropna(subset=[1], inplace=True)

    #Printing classs distribuiton
    print(data[1].value_counts())

    data[0] = data[0].apply(preprocessing)
    data.dropna(inplace=True)

    data.to_csv(SAVE_PATH, index=False, header=False)
    print(f"Saved to: {SAVE_PATH}")

if __name__=="__main__":
    main()
    exit(0)
#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<math.h>

using namespace std;

int main(){
    ifstream inFile("data2_original.dat");
    string line;
    vector<float> vec;
    vec.resize(200);
    vector<float> loss(200), acc(200), val_loss(200), val_acc(200);
    size_t sz;
    int eps=200;
    if(inFile.is_open()){
        int i=0;
        while (getline(inFile,line)){
            int begin=0;
            size_t pos=line.find("step");
            string temp;
            if (pos!=std::string::npos){
                pos=line.find("loss:");
                pos+=6;
                temp=line.substr(pos,6);
                loss[i]=stof(temp, &sz);

                pos=line.find("acc:");
                pos+=5;
                temp=line.substr(pos,6);
                acc[i]=stof(temp, &sz);

                pos=line.find("val_loss:");
                pos+=10;
                temp=line.substr(pos,6);
                val_loss[i]=stof(temp, &sz);

                pos=line.find("val_acc:");
                pos+=9;
                temp=line.substr(pos,6);
                val_acc[i]=stof(temp, &sz);
                
                i+=1;
            }
        }
    }
    ofstream outFile0("data2.dat");
    for (int j=0; j<eps; ++j)
        outFile0<<j+1<<' '<<loss[j]<<' '<<acc[j]<<' '<< 
                        val_loss[j]<<' '<<val_acc[j]<<endl;
    outFile0.close();

    return 0;
}

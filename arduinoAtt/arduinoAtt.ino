#include <AESLib.h>

char data[20];

void setup() {
    // put your setup code here, to run once
    Serial.begin(9600);
    Serial.setTimeout(1);
    /*
    */
    pinMode(12, OUTPUT);
    pinMode(11, OUTPUT); 
}

void printInputInInt()
{   
    int i;
    for (i=0;i<16;i++){
        Serial.print((int)data[i]);
        Serial.print(", ");
    } 
    Serial.println("");
}

void getSerailData()
{
    String rez;
    String x;
    bool getAllData = false;
    int i = 0;
    while(!getAllData){
        while (!Serial.available());
        x = Serial.readString();
        if(x == '\0')  getAllData = true;
        else rez+=x;
    } 
    Serial.println(rez);
    rez.toCharArray(data, 20); //16 chars == 16 bytes
    /* printInputInInt(); */
}

void loop() {
  // put your main code here, to run repeatedly:
    getSerailData();
    /* char data[] = "0123456789012345"; */
    uint8_t key[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    PORTB |= 0x10; 
    aes128_enc_single(key, data);
    PORTB &= ~0x10; 
    aes128_dec_single(key, data);
    Serial.print("decrypted:");
    Serial.print(data);
}

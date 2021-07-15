int flexs=A0;
int data = 0; 


void setup()
{
  Serial.begin(9600); 
  pinMode(flexs, INPUT);   
}
 
void loop()
{
  data = analogRead(flexs); 
  Serial.println(data); 
  delay(1000); 
}

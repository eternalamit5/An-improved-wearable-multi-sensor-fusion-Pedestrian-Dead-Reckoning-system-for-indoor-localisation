int flexs=A0;
int data = 0; 

unsigned long sampleTimer = 0;
unsigned long sampleInterval = 0.1; //0.1 ms = 100Hz rate
 
//int led = 13; 

void setup()
{
  Serial.begin(115200); 
  pinMode(flexs, INPUT);
//pinMode(led, OUTPUT);   
}
 
void loop()
{
  unsigned long currMillis = millis();
  if(currMillis - sampleTimer >= sampleInterval)  // is it time for a sample?
  {
    sampleTimer = currMillis; 
   data = analogRead(flexs); 
  Serial.println(data); 
  }
}



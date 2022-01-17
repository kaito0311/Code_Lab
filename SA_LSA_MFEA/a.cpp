#include <stdio.h>
int main(){
    char S[1000], T[1000]; 
    scanf("%s", S); 
    scanf("%s", T); 
    
    int ascii[257] = {0};
    int i = 0; 
    for(i = 0; S[i] != '\0'; i++)
    {
        ascii[(int)(S[i])] += 1; 
    }
    for(i = 0; T[i] != '\0'; i++){
        ascii[(int)(T[i])] += 1;   
    }
    
    for(i = 0; i < 256; i++){
        if(ascii[i]!= 0){
            printf("%c", i);
        }
    }
    printf("\n");
    for(i = 0; i < 256; i++){
        if(ascii[i]!= 0){
            printf("%d\n", ascii[i]);
        }
    }
    
}
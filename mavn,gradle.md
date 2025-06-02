coommands for the mavean project
1. mvn compile
2. mvn test
3. mvn package
4. mvn clean install
5. java -cp target/myapp-1.0-SNAPSHOT.jar com.example.App

commands for the gradle project
1. gradle build
2. gradle run


convert the maven to gradle projet
1. create one mvn project and run the following command to the generate the jar file
 1. Execute the following command:
java -cp target/HelloMaven-1.0-SNAPSHOT.jar com.example.App
 here com.example.App, file replace your actual directory path.

now convert the mvn to gradle
1. gradle init
 after theis command they will ask the some input then give there inputs and after  edit the "build.gradle"
"application {
// Update the mainClass to reflect the package structure
mainClass = 'com.example.App'
}"

after editing the run the following the command
gradle build
gradle run

plugins {
    kotlin("jvm") version "2.0.20"
}

group = "com.lignting"
version = "1.0-SNAPSHOT"

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
    implementation("org.jetbrains.kotlinx:dataframe:0.15.0")
    implementation("org.jetbrains.kotlinx:kandy-lets-plot:0.7.1")
    implementation(project(":kTouch"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(18)
}
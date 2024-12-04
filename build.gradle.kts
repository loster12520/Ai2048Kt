plugins {
    kotlin("jvm") version "2.0.20"
}

group = "com.lignting"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(18)
}
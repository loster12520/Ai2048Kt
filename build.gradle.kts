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

    implementation("org.jfree:jfreechart:1.5.3")
    implementation("org.jfree:jcommon:1.0.24")
    implementation("com.jsoizo:kotlin-csv-jvm:1.10.0")
    implementation("org.jetbrains.kotlinx:dataframe:0.15.0")
    implementation("org.jetbrains.kotlinx:kandy-lets-plot:0.7.1")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(18)
}
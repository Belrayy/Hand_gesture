fun sendMessageToClient(
        client: Client?, message: String?, mailer: Mailer
) {
    val mail=client?.personalInfo?.email
    if(mail != null && message != null){
        mailer.sendMessage(mail,message)
    }
}

class Client(val personalInfo: PersonalInfo?)
class PersonalInfo(val email: String?)
interface Mailer {
    fun sendMessage(email: String, message: String)
}
